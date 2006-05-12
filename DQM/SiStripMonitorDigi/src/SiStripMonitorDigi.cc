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
// $Id: SiStripMonitorDigi.cc,v 1.7 2006/05/03 08:39:01 dkcira Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/Common/interface/DetSetVector.h" // replaces SiStripDigiCollection
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

using namespace std;
using namespace edm;

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
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_mechanical_structure_view = "<<show_mechanical_structure_view;
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_readout_view = "<<show_readout_view;
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_control_view = "<<show_control_view;

  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    es.get<SiStripDetCablingRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripDetCabling
    const vector<uint32_t> & activeDets = tkmechstruct->getActiveDetectorsRawIds();

    // use SiStripSubStructure for selecting certain regions
    SiStripSubStructure substructure;
    vector<uint32_t> SelectedDetIds;
    substructure.getTIBDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
//    substructure.getTOBDetectors(activeDets, SelectedDetIds, 1, 2, 0);    // this adds rawDetIds to SelectedDetIds
//    substructure.getTIDDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
//    substructure.getTECDetectors(activeDets, SelectedDetIds, 1, 2, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds

     // use SistripHistoId for producing histogram id (and title)
     SiStripHistoId hidmanager;
     // create SiStripFolderOrganizer
     SiStripFolderOrganizer folder_organizer;

    // loop over detectors and book MEs
    LogInfo("SiStripTkDQM")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
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
      hid = hidmanager.createHistoId("ADCsHottest strip","det",*detid_iterator);
      local_modmes.ADCsHottestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.);
      // create ADCs per "coolest" strip
      hid = hidmanager.createHistoId("ADCsCoolest strip","det",*detid_iterator);
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

  // retrieve producer name of input StripDigiCollection
  std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
  // get collection of DetSetVector of digis from Event
  edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor;
  iEvent.getByLabel(digiProducer, digi_detsetvektor);
  // loop over all MEs
  for (map<uint32_t, ModMEs >::const_iterator iterMEs = DigiMEs.begin() ; iterMEs!=DigiMEs.end() ; iterMEs++) {
    uint32_t detid = iterMEs->first; ModMEs local_modmes = iterMEs->second; // get detid and type of ME
    // get from DetSetVector the DetSet of digis belonging to one detid - first make sure there exists digis with this id
    edm::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detsetvektor->find(detid); // search  digis of detid
    if(isearch==digi_detsetvektor->end()) continue; // no digis for this detid => jump to next step of loop
    //digi_detset is a structure, digi_detset.data is a std::vector<SiStripDigi>, digi_detset.id is uint32_t
    edm::DetSet<SiStripDigi> digi_detset = (*digi_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
    //
    if(local_modmes.DigisPerModule != NULL){ // nr. of digis per detector
      (local_modmes.DigisPerModule)->Fill(static_cast<float>(digi_detset.data.size()),1.);
    }
    if(local_modmes.ADCsHottestStrip != NULL){ // nr. of adcs for hottest strip
      int largest_adc=(digi_detset.data.begin())->adc();
      for(edm::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.data.begin(); digiIter!= digi_detset.data.end(); digiIter++ ){
           if(digiIter->adc()>largest_adc) largest_adc = digiIter->adc(); 
      }
      (local_modmes.ADCsHottestStrip)->Fill(static_cast<float>(largest_adc),1.);
    }
    if(local_modmes.ADCsCoolestStrip){ // nr. of adcs for coolest strip
      int smallest_adc=(digi_detset.data.begin())->adc();
      for(edm::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.data.begin(); digiIter!= digi_detset.data.end(); digiIter++ ){
           if(digiIter->adc()<smallest_adc) smallest_adc = digiIter->adc(); 
      }
      (local_modmes.ADCsCoolestStrip)->Fill(static_cast<float>(smallest_adc),1.);
    }
  }
}


void SiStripMonitorDigi::endJob(void){
   bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
   string outputFileName = conf_.getParameter<string>("OutputFileName");
//  dbe_->showDirStructure();
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi)
