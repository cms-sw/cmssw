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
// $Id: SiStripMonitorDigi.cc,v 1.15 2006/10/27 01:35:21 wmtan Exp $
//
//

#include<fstream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"


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
   bool select_all_detectors = conf_.getParameter<bool>("SelectAllDetectors");
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowMechanicalStructureView = "<<show_mechanical_structure_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowReadoutView = "<<show_readout_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowControlView = "<<show_control_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"SelectAllDetectors = "<<select_all_detectors;


  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    es.get<SiStripDetCablingRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripDetCabling
    std::vector<uint32_t> activeDets; 
    activeDets.clear(); // just in case
    tkmechstruct->addActiveDetectorsRawIds(activeDets);

    std::vector<uint32_t> SelectedDetIds;
    if(select_all_detectors){
      // select all detectors if appropriate flag is set,  for example for the mtcc
      SelectedDetIds = activeDets;
    }else{
      // use SiStripSubStructure for selecting certain regions
      SiStripSubStructure substructure;
      substructure.getTIBDetectors(activeDets, SelectedDetIds, 1, 1, 1, 0); // this adds rawDetIds to SelectedDetIds
//      substructure.getTOBDetectors(activeDets, SelectedDetIds, 1, 2, 0);    // this adds rawDetIds to SelectedDetIds
//      substructure.getTIDDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
//      substructure.getTECDetectors(activeDets, SelectedDetIds, 1, 2, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds
    }

     // remove any eventual zero elements - there should be none, but just in case
     for(std::vector<uint32_t>::iterator idets = SelectedDetIds.begin(); idets != SelectedDetIds.end(); idets++){
       if(*idets == 0) SelectedDetIds.erase(idets);
     }

     // use SistripHistoId for producing histogram id (and title)
     SiStripHistoId hidmanager;
     // create SiStripFolderOrganizer
     SiStripFolderOrganizer folder_organizer;

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
    for(std::vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs local_modmes;
      std::string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
//
//      // create ADCs per strip
//      std::string hid = hidmanager.createHistoId("ADCsPerStrip_detector", *detid_iterator);
//      local_me = dbe_->book2D(hid, hid, 20,-0.5,767.5, 20,-0.5,255.5);
//      ADCsPerStrip.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
//
      // create Digis per detector - not too useful - maybe can remove later
      hid = hidmanager.createHistoId("NumberOfDigis","det",*detid_iterator);
      local_modmes.NumberOfDigis = dbe_->book1D(hid, hid, 21, -0.5, 20.5); dbe_->tag(local_modmes.NumberOfDigis, *detid_iterator);
      // create ADCs per "hottest" strip
      hid = hidmanager.createHistoId("ADCsHottestStrip","det",*detid_iterator);
      local_modmes.ADCsHottestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.); dbe_->tag(local_modmes.ADCsHottestStrip, *detid_iterator);
      // create ADCs per "coolest" strip
      hid = hidmanager.createHistoId("ADCsCoolestStrip","det",*detid_iterator);
      local_modmes.ADCsCoolestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.); dbe_->tag(local_modmes.ADCsCoolestStrip, *detid_iterator);
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

//  // retrieve producer name of input StripDigiCollection
//  std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
//  std::string digiLabel    = conf_.getParameter<std::string>("DigiLabel");
//  // get collection of DetSetVector of digis from Event
//  edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor;
//  iEvent.getByLabel(digiProducer, digiLabel, digi_detsetvektor);

// get all digi collections
   edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor;
   typedef std::vector<edm::ParameterSet> Parameters;
   Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
   Parameters::iterator itDigiProducersList = DigiProducersList.begin();
   for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {
   std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
   std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
   iEvent.getByLabel(digiProducer,digiLabel,digi_detsetvektor);

   // loop over all MEs
    for (std::map<uint32_t, ModMEs >::const_iterator iterMEs = DigiMEs.begin() ; iterMEs!=DigiMEs.end() ; iterMEs++) {
      uint32_t detid = iterMEs->first; ModMEs local_modmes = iterMEs->second; // get detid and type of ME
      // get from DetSetVector the DetSet of digis belonging to one detid - first make sure there exists digis with this id
      edm::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detsetvektor->find(detid); // search  digis of detid
      if(isearch==digi_detsetvektor->end()) continue; // no digis for this detid => jump to next step of loop
      //digi_detset is a structure, digi_detset.data is a std::vector<SiStripDigi>, digi_detset.id is uint32_t
      edm::DetSet<SiStripDigi> digi_detset = (*digi_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
      //
      if(local_modmes.NumberOfDigis != NULL){ // nr. of digis per detector
        (local_modmes.NumberOfDigis)->Fill(static_cast<float>(digi_detset.data.size()),1.);
      }
      if(digi_detset.data.size()==0) continue; // continue with next ME if 0 digis in this detset
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
}

void SiStripMonitorDigi::endJob(void){
   bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
   std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    std::ofstream monitor_summary("monitor_digi_summary.txt");
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    monitor_summary<<"SiStripMonitorDigi::endJob DigiMEs.size()="<<DigiMEs.size()<<std::endl;
    for(std::map<uint32_t, ModMEs>::const_iterator idet = DigiMEs.begin(); idet!= DigiMEs.end(); idet++ ){
     monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"      ++++++detid  "<<idet->first<<std::endl<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ NumberOfDigis "<<(idet->second).NumberOfDigis->getEntries()<<" "<<(idet->second).NumberOfDigis->getMean()<<" "<<(idet->second).NumberOfDigis->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ ADCsHottestStrip "<<(idet->second).ADCsHottestStrip->getEntries()<<" "<<(idet->second).ADCsHottestStrip->getMean()<<" "<<(idet->second).ADCsHottestStrip->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ ADCsCoolestStrip "<<(idet->second).ADCsCoolestStrip->getEntries()<<" "<<(idet->second).ADCsCoolestStrip->getMean()<<" "<<(idet->second).ADCsCoolestStrip->getRMS()<<std::endl;
    }
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    // save histograms in a file
    dbe_->save(outputFileName);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi);
