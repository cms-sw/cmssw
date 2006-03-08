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
// $Id: SiStripMonitorDigi.cc,v 1.1 2006/02/09 19:08:43 gbruno Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h" // these two will go away
#include "CalibTracker/Records/interface/SiStripStructureRcd.h"     // these two will go away

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
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

    // use SiStripSubStructure for selecting certain regions
    SiStripSubStructure substructure;
    vector<uint32_t> SelectedDetIds;
    // select TIBs of layer=2. 0 selects everything
    substructure.getTIBDetectors(activeDets, SelectedDetIds, 2, 7, 2, 0); // this adds rawDetIds to SelectedDetIds
    // select TOBs of layer=1, etc.
    substructure.getTOBDetectors(activeDets, SelectedDetIds, 1, 3, 4, 0); // this adds rawDetIds to SelectedDetIds

     // use SistripHistoId for producing histogram id (and title)
     SiStripHistoId hidmanager;
     // create SiStripFolderOrganizer
     SiStripFolderOrganizer folder_organizer;

    // loop over detectors and book MEs
    cout<<"SiStripMonitorDigis::analyze nr. of SelectedDetIds:  "<<SelectedDetIds.size()<<endl;
    for(vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs local_modmes;
      string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
//
//      // create ADCs per strip
//      string hid = hidmanager.createHistoId("ADCsPerStrip_detector", *detid_iterator);
//      local_me = dbe_->book2D(hid, hid, 20,-0.5,767.5, 20,-0.5,255.5);
//      ADCsPerStrip.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
//
      // create Digis per detector - not too useful - maybe can remove later
      hid = hidmanager.createHistoId("DigisPerDetector","det",*detid_iterator);
      local_modmes.DigisPerModule = dbe_->book1D(hid, hid, 21, -0.5, 20.5);
      // create ADCs per "hottest" strip
      hid = hidmanager.createHistoId("ADCs of hottest strip","det",*detid_iterator);
      local_modmes.ADCsHottestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.);
      // create ADCs per "coolest" strip
      hid = hidmanager.createHistoId("ADCs of coolest strip","det",*detid_iterator);
      local_modmes.ADCsCoolestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.);
      // append to DigiMEs
      DigiMEs.insert( std::make_pair(*detid_iterator, local_modmes));
      //
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

  // loop over all MEs
  for (map<uint32_t, ModMEs >::const_iterator i = DigiMEs.begin() ; i!=DigiMEs.end() ; i++) {
    uint32_t detid = i->first; ModMEs local_modmes = i->second;

    // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
    // get DigiCollection object from Event
    edm::Handle<StripDigiCollection> digi_collection;
    iEvent.getByLabel(digiProducer, digi_collection);
    // get iterators for digis belonging to one DetId
    const StripDigiCollection::Range digiRange = digi_collection->get(detid);
    StripDigiCollection::ContainerIterator digiBegin = digiRange.first;
    StripDigiCollection::ContainerIterator digiEnd   = digiRange.second;

    if(local_modmes.DigisPerModule != NULL){ // nr. of digis per detector
      (local_modmes.DigisPerModule)->Fill(static_cast<float>(digiEnd-digiBegin),1.);
    }
    if(local_modmes.ADCsHottestStrip != NULL){ // nr. of adcs for hottest strip
      int largest_adc=digiBegin->adc();
      for(StripDigiCollection::ContainerIterator digiIter = digiBegin; digiIter<digiEnd; digiIter++){
           if(digiIter->adc()>largest_adc) largest_adc = digiIter->adc(); 
      }
      (local_modmes.ADCsHottestStrip)->Fill(static_cast<float>(largest_adc),1.);
    }
    if(local_modmes.ADCsCoolestStrip){ // nr. of adcs for coolest strip
      int smallest_adc=digiBegin->adc();
      for(StripDigiCollection::ContainerIterator digiIter = digiBegin; digiIter<digiEnd; digiIter++){
           if(digiIter->adc()<smallest_adc) smallest_adc = digiIter->adc(); 
      }
      (local_modmes.ADCsCoolestStrip)->Fill(static_cast<float>(smallest_adc),1.);
    }
  }
}


void SiStripMonitorDigi::endJob(void){
//  dbe_->showDirStructure();
//  dbe_->save("test_digi.root");
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi)
