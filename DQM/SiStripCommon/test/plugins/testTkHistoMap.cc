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
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"


#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include <math.h>
#include <vector>
#include <sstream>
using std::cout; using std::endl; using std::string;


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
    
  TkHistoMap *tkhisto, *tkhistoZ, *tkhistoPhi, *tkhistoR;
     
};
//
testTkHistoMap::testTkHistoMap ( const edm::ParameterSet& iConfig )
{
  tkhisto   =new TkHistoMap("pippo","pluto");
  tkhistoZ  =new TkHistoMap("Z","Z");
  tkhistoPhi=new TkHistoMap("Phi","Phi");
  tkhistoR  =new TkHistoMap("R","R");
}


testTkHistoMap::~testTkHistoMap()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

void testTkHistoMap::endJob(void)
{
  edm::Service<DQMStore>().operator->()->save("test.root");  
}


//
// member functions
//

// // ------------ method called to produce the data  ------------
void testTkHistoMap::analyze(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup )
{   
  edm::ESHandle<TrackerGeometry> tkgeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );

  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  std::vector<uint32_t> TkDetIdList,fullTkDetIdList=fr->getAllDetIds();
  float value;

  //TkDetIdList=fullTkDetIdList;

  SiStripSubStructure siStripSubStructure;
  //extract  vector of module in the layer
  siStripSubStructure.getTOBDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
  /*  
  for(size_t i=0;i<TkDetIdList.size();++i){
    value = TkDetIdList[i]%1000000;
    //    tkhisto->fill(TkDetIdList[i],value);
    tkhisto->setBinContent(TkDetIdList[i],value);
  }%
  */

  LocalPoint localPos(0.,0.,0.);
  GlobalPoint globalPos;
  
  for(size_t i=0;i<TkDetIdList.size();++i){

    const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(TkDetIdList[i])));
    globalPos=(_StripGeomDetUnit->surface()).toGlobal(localPos);
    
    std::cout << "detid " << TkDetIdList[i] << " pos z " << globalPos.z() << " phi " << globalPos.phi() << " r " << globalPos.perp()<<std::endl;;
    value = TkDetIdList[i]%1000000;


    //    tkhisto->fill(TkDetIdList[i],value);
    tkhisto->fill(TkDetIdList[i],value);
    tkhistoZ->fill(TkDetIdList[i],globalPos.z());
    tkhistoPhi->fill(TkDetIdList[i],globalPos.phi());
    tkhistoR->fill(TkDetIdList[i],globalPos.perp());
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(testTkHistoMap);
