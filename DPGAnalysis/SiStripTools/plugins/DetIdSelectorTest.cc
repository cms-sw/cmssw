// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      DetIdSelectorTest
// 
/**\class DetIdSelectorTest DetIdSelectorTest.cc DPGAnalysis/SiStripTools/plugins/DetIdSelectorTest.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jul 19 11:56:00 CEST 2009
//
//


// system include files
#include <memory>

// user include files
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
//***************************************************

//
// class decleration
//

class DetIdSelectorTest : public edm::EDAnalyzer {
public:
  explicit DetIdSelectorTest(const edm::ParameterSet&);
  ~DetIdSelectorTest();


private:
  virtual void beginJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

      // ----------member data ---------------------------


  std::vector<DetIdSelector> detidsels_;
  TkHistoMap *tkhisto_;
  TrackerMap tkmap_;

  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DetIdSelectorTest::DetIdSelectorTest(const edm::ParameterSet& iConfig):
  detidsels_(), tkhisto_(new TkHistoMap("SelectorTest","SelectorTest",-1)),tkmap_()
{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet> >("selections");

  for(std::vector<edm::ParameterSet>::const_iterator selconfig=selconfigs.begin();selconfig!=selconfigs.end();++selconfig) {
    DetIdSelector selection(*selconfig);
    detidsels_.push_back(selection);
  }

  tkmap_.setPalette(1);
  tkmap_.addPixel(true);

}


DetIdSelectorTest::~DetIdSelectorTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DetIdSelectorTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   {
     SiStripDetInfoFileReader * reader=edm::Service<SiStripDetInfoFileReader>().operator->();
     
     //   SiStripDetInfoFileReader reader(edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
     //   SiStripDetInfoFileReader reader;
     
     const std::vector<uint32_t>& detids = reader->getAllDetIds();
     
     for(std::vector<uint32_t>::const_iterator detid=detids.begin();detid!=detids.end();++detid) {
       
       LogDebug("DetID") << *detid;
       int index=0;
       for(std::vector<DetIdSelector>::const_iterator detidsel=detidsels_.begin();detidsel!=detidsels_.end();++detidsel) {
	 if(detidsel->isSelected(*detid)) {
	   LogDebug("selected") << "Selected by selection " << index;
	   unsigned int det = *detid;
	   tkhisto_->add(det,index);
	   tkmap_.fill_current_val(det,index);
	 }
	 ++index;
       }
       
     }
   }

   {
     edm::FileInPath fp("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt");
     
     SiPixelDetInfoFileReader pxlreader(fp.fullPath());
     const std::vector<uint32_t>& detids = pxlreader.getAllDetIds();
     
     for(std::vector<uint32_t>::const_iterator detid=detids.begin();detid!=detids.end();++detid) {
       
       LogDebug("DetID") << *detid;
       int index=0;
       for(std::vector<DetIdSelector>::const_iterator detidsel=detidsels_.begin();detidsel!=detidsels_.end();++detidsel) {
	 if(detidsel->isSelected(*detid)) {
	   LogDebug("selected") << "Selected by selection " << index;
	   unsigned int det = *detid;
	   //	   tkhisto_->add(det,index);
	   tkmap_.fill_current_val(det,index);
	 }
	 ++index;
       }
       
     }
   }
    
 
   /*
     edm::ESHandle<TrackerGeometry> pDD;
     iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
     
     for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
     
     if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
     DetId detId = (*it)->geographicalId();
     LogDebug("DetID") << detId.rawId();
     int index=0;
     for(std::vector<DetIdSelector>::const_iterator detidsel=detidsels_.begin();detidsel!=detidsels_.end();++detidsel) {
     if(detidsel->isSelected(detId)) {
     LogDebug("selected") << " Selected by selection " << index;
     //	 tkhisto_->add(det,index);
     tkmap_.fill_current_val(detId.rawId(),index);
     }
     ++index;
     }
     
     }
     
     }
   */   
}

void 
DetIdSelectorTest::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{}

void 
DetIdSelectorTest::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
DetIdSelectorTest::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
DetIdSelectorTest::endJob() {


  //  tkhisto_->dumpInTkMap(&tkmap);
  std::string mapname = "SelectorTest.png";
  tkmap_.save(true,0,0,mapname);

  std::string rootmapname = "TKMap_Selectortest.root";
  tkhisto_->save(rootmapname);
}


//define this as a plug-in
DEFINE_FWK_MODULE(DetIdSelectorTest);
