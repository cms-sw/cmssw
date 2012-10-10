// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"


#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "TPostScript.h"
#include "TCanvas.h"

#include <math.h>
#include <vector>
#include <sstream>

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
//***************************************************


//
// class declaration
//

class testTkHistoMap : public edm::EDAnalyzer {
public:
  explicit testTkHistoMap ( const edm::ParameterSet& );
  ~testTkHistoMap ();
   
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  virtual void endJob(void);

private:
  
  void read();
  void create(); 

  bool readFromFile;
  TkHistoMap *tkhisto, *tkhistoBis, *tkhistoZ, *tkhistoPhi, *tkhistoR, *tkhistoCheck;
};

//
testTkHistoMap::testTkHistoMap ( const edm::ParameterSet& iConfig ):
  readFromFile(iConfig.getParameter<bool>("readFromFile"))
{
  if(!readFromFile){
    create();
  }else{
    read();
  }
}


void testTkHistoMap::create(){
  tkhisto      =new TkHistoMap("detId","detId",-1); 
  tkhistoBis   =new TkHistoMap("detIdBis","detIdBis",0,1); //here the baseline (the value of the empty,not assigned bins) is put to -1 (default is zero)
  tkhistoZ     =new TkHistoMap("Zmap","Zmap");
  tkhistoPhi   =new TkHistoMap("Phi","Phi");
  tkhistoR     =new TkHistoMap("Rmap","Rmap",-99.); //here the baseline (the value of the empty,not assigned bins) is put to -99 (default is zero)
  tkhistoCheck =new TkHistoMap("check","check");           
}

/*Check that is possible to load in tkhistomaps histograms already stored in a DQM root file (if the folder and name are know)*/
void testTkHistoMap::read(){
  edm::Service<DQMStore>().operator->()->open("test.root");  

  tkhisto      =new TkHistoMap();
  tkhistoBis   =new TkHistoMap();
  tkhistoZ     =new TkHistoMap();
  tkhistoPhi   =new TkHistoMap();
  tkhistoR     =new TkHistoMap();
  tkhistoCheck =new TkHistoMap();

  tkhisto     ->loadTkHistoMap("detId","detId"); 	    
  tkhistoBis  ->loadTkHistoMap("detIdBis","detIdBis",1);  
  tkhistoZ    ->loadTkHistoMap("Zmap","Zmap");		    
  tkhistoPhi  ->loadTkHistoMap("Phi","Phi");		    
  tkhistoR    ->loadTkHistoMap("Rmap","Rmap");
  tkhistoCheck->loadTkHistoMap("check","check");            
}

testTkHistoMap::~testTkHistoMap()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

void testTkHistoMap::endJob(void)
{
  /*Test extraction of detid from histogram title and ix, iy*/
  size_t ilayer=1;
  std::string histoTitle=tkhisto->getMap(ilayer)->getTitle();
  uint32_t detB=tkhisto->getDetId(ilayer,5,5);
  uint32_t detA=tkhisto->getDetId(histoTitle,5,5);
  if(detA==0 || detA!=detB)
    edm::LogError("testTkHistoMap") << " for layer " << ilayer << " the extracted detid in a bin is wrong " << detA << " " << detB << std::endl;


  /*Test Drawing functions*/
  TCanvas C("c","c");
  C.Divide(3,3);
  C.Update(); 
  TPostScript ps("test.ps",121);
  ps.NewPage();
  for(size_t ilayer=1;ilayer<34;++ilayer){
    C.cd(1);
    tkhisto->getMap   (ilayer)->getTProfile2D()->Draw("TEXT");
    C.cd(2);
    tkhistoZ->getMap  (ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(3);
    tkhistoPhi->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(4);
    tkhistoR->getMap  (ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(5);
    tkhistoCheck->getMap  (ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(6);
    tkhistoBis->getMap  (ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.Update();
    ps.NewPage();
  }
  ps.Close();   

  if(!readFromFile)
    edm::Service<DQMStore>().operator->()->save("test.root");  
  
  tkhisto->saveAsCanvas("test.canvas.root","LEGO","RECREATE");
  tkhistoBis->saveAsCanvas("test.canvas.root","LEGO","RECREATE");
  tkhistoZ->saveAsCanvas("test.canvas.root","LEGO","UPDATE");
  tkhistoPhi->saveAsCanvas("test.canvas.root","LEGO","UPDATE");
  tkhistoR->saveAsCanvas("test.canvas.root","LEGO","UPDATE");
  tkhistoCheck->saveAsCanvas("test.canvas.root","LEGO","UPDATE");
  
  /* test Dump in TkMap*/
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
  TrackerMap tkmap, tkmapZ, tkmapPhi, tkmapR; 

  tkmap.setPalette(1);
  tkmapZ.setPalette(2);
  tkmapPhi.setPalette(2);
  tkmapR.setPalette(2);
  
  tkhisto->dumpInTkMap(&tkmap);
  tkhistoZ->dumpInTkMap(&tkmapZ);
  tkhistoPhi->dumpInTkMap(&tkmapPhi);
  tkhistoR->dumpInTkMap(&tkmapR);

  tkmap.save(true,0,0,"testTkMap.png");
  tkmapZ.save(true,0,0,"testTkMapZ.png");
  tkmapPhi.save(true,0,0,"testTkMapPhi.png");
  tkmapR.save(true,0,0,"testTkMapR.png");
}


//
// member functions
//

// // ------------ method called to produce the data  ------------
void testTkHistoMap::analyze(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup )
{   
  if(readFromFile)
    return;

  edm::ESHandle<TrackerGeometry> tkgeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );

  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  std::vector<uint32_t> TkDetIdList,fullTkDetIdList=fr->getAllDetIds();
  float value;
  LocalPoint localPos(0.,0.,0.);
  GlobalPoint globalPos;

  TkDetIdList=fullTkDetIdList;

  //extract  vector of module in the layer
  /*
    SiStripSubStructure siStripSubStructure;
    siStripSubStructure.getTIBDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
    siStripSubStructure.getTOBDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
    siStripSubStructure.getTIDDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
    //siStripSubStructure.getTECDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
  */

  tkhisto->fillFromAscii("test.txt");
  tkhistoBis->fillFromAscii("test2.txt");

  for(size_t i=0;i<TkDetIdList.size();++i){

    const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(TkDetIdList[i])));
    globalPos=(_StripGeomDetUnit->surface()).toGlobal(localPos);
    
    value = TkDetIdList[i]%1000000;
    
    //tkhisto->fill(TkDetIdList[i],value);
    //tkhistoBis->fill(TkDetIdList[i],value);
    tkhistoZ->fill(TkDetIdList[i],globalPos.z());
    tkhistoPhi->fill(TkDetIdList[i],globalPos.phi());
    tkhistoR->fill(TkDetIdList[i],globalPos.perp());
    tkhistoCheck->add(TkDetIdList[i],1.);
    tkhistoCheck->add(TkDetIdList[i],1.);

    edm::LogInfo("testTkHistoMap") << "detid " << TkDetIdList[i] << " pos z " << globalPos.z() << " phi " << globalPos.phi() << " r " << globalPos.perp() << std::endl;

    if(value!=tkhisto->getValue(TkDetIdList[i]))
      edm::LogError("testTkHistoMap") << " input value " << value << " differs from read value " << tkhisto->getValue(TkDetIdList[i]) << std::endl;

    // For usage that reset histo content use setBinContent instead than fill
    /* 
    tkhisto->setBinContent(TkDetIdList[i],value);
    tkhistoZ->setBinContent(TkDetIdList[i],globalPos.z());
    tkhistoPhi->setBinContent(TkDetIdList[i],globalPos.phi());
    tkhistoR->setBinContent(TkDetIdList[i],globalPos.perp());
    */
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(testTkHistoMap);
