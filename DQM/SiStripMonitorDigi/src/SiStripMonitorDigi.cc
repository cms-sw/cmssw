// -*- C++ -*-
//
// Package:    SiStripMonitorDigi
// Class:      SiStripMonitorDigi
// 
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dorian Kcira
//         Created:  Sat Feb  4 20:49:10 CET 2006
// $Id$
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h" // these two will go away
#include "CalibTracker/Records/interface/SiStripStructureRcd.h"     // these two will go away

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoIdManager.h"
#include "DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

using namespace std;

SiStripMonitorDigi::SiStripMonitorDigi(const edm::ParameterSet& iConfig)
{
   dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
   conf_ = iConfig;
}


SiStripMonitorDigi::~SiStripMonitorDigi()
{
}


void SiStripMonitorDigi::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file
   bool show_mechanical_structure_view = conf_.getParameter<bool>("ShowMechanicalStructureView");
   bool show_readout_view = conf_.getParameter<bool>("ShowReadoutView");
   bool show_control_view = conf_.getParameter<bool>("ShowControlView");
   cout<<"SiStripMonitorDigi::beginJob : show_mechanical_structure_view = "<<show_mechanical_structure_view<<endl;
   cout<<"SiStripMonitorDigi::beginJob : show_readout_view = "<<show_readout_view<<endl;
   cout<<"SiStripMonitorDigi::beginJob : show_control_view = "<<show_control_view<<endl;

  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripStructure object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripStructure> tkmechstruct;
    es.get<SiStripStructureRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl object
    const vector<uint32_t> & activeDets = tkmechstruct->getActiveDetectorsRawIds();

    // select (certain) TOB detectors from activeDets and put them in TOBDetIds
    vector<uint32_t> TOBDetIds;
    SiStripSubStructure substructure;
    // select TIBs of layer=2. 0 selects everything
    // substructure.getTIBDetectors(activeDets, TIBDetIds, 2, 0, 0, 0);
    // select TOBs of layer=1, etc.
    substructure.getTOBDetectors(activeDets, TOBDetIds, 1, 3, 4, 0);

     // use SistripHistoIdManager for producing histogram id (and title)
     SiStripHistoIdManager hidmanager;
    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;
    MonitorElement* local_me;

    // loop over detectors and book MEs
    cout<<"SiStripMonitorDigis::analyze nr. of TOBDetIds:  "<<TOBDetIds.size()<<endl;
    for(vector<uint32_t>::const_iterator detid_iterator = TOBDetIds.begin(); detid_iterator!=TOBDetIds.end(); detid_iterator++){
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
//      // create ADCs per strip
//      string hid = hidmanager.createHistoId("ADCsPerStrip_detector", *detid_iterator);
//      local_me = dbe_->book2D(hid, hid, 20,-0.5,767.5, 20,-0.5,255.5);
//      ADCsPerStrip.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
      // create ADCs per detector
      string hid = hidmanager.createHistoId("DigisPerDetector", *detid_iterator);
      local_me = dbe_->book1D(hid, hid, 21, -0.5, 20.5);
      DigisPerDetector.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
    }
  }
}


// ------------ method called to produce the data  ------------
void
SiStripMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

  // Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
//  for (map<uint32_t, MonitorElement*>::const_iterator i = ADCsPerStrip.begin() ; i!=ADCsPerStrip.end() ; i++) {
//    uint32_t detid = i->first;
//    MonitorElement* local_me = i->second;
//
//    // retrieve producer name of input StripDigiCollection
//    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
//    // get DigiCollection object from Event
//    edm::Handle<StripDigiCollection> digi_collection;
//    iEvent.getByLabel(digiProducer, digi_collection);
//
//    // get iterators for digis belonging to one DetId
//    const StripDigiCollection::Range digiRange = digi_collection->get(detid);
//    StripDigiCollection::ContainerIterator digiBegin = digiRange.first;
//    StripDigiCollection::ContainerIterator digiEnd   = digiRange.second;
//
//    // loop over Digis belonging to detid
//    for(StripDigiCollection::ContainerIterator digiIter = digiBegin; digiIter<digiEnd; digiIter++){
//      // fill ME
//      local_me->Fill(digiIter->strip(),digiIter->adc(),1.);
//    }
//  }

  for (map<uint32_t, MonitorElement*>::const_iterator i = DigisPerDetector.begin() ; i!=DigisPerDetector.end() ; i++) {
    uint32_t detid = i->first;
    MonitorElement* local_me = i->second;

    // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
    // get DigiCollection object from Event
    edm::Handle<StripDigiCollection> digi_collection;
    iEvent.getByLabel(digiProducer, digi_collection);

    // get iterators for digis belonging to one DetId
    const StripDigiCollection::Range digiRange = digi_collection->get(detid);
    StripDigiCollection::ContainerIterator digiBegin = digiRange.first;
    StripDigiCollection::ContainerIterator digiEnd   = digiRange.second;

    local_me->Fill(static_cast<float>(digiEnd-digiBegin),1.);
  }

}


void SiStripMonitorDigi::endJob(void){
//  dbe_->showDirStructure();
//  dbe_->save("test_digi.root");
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi)
