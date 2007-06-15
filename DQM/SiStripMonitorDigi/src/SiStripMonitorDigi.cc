// -*- c++ -*-
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc
*/
// Original Author:  Dorian Kcira
//         Created:  Sat Feb  4 20:49:10 CET 2006
// $Id: SiStripMonitorDigi.cc,v 1.20 2007/06/08 14:50:53 dkcira Exp $
#include<fstream>
#include "TNamed.h"
#include "TH1F.h"
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
#include "DQMServices/Core/interface/MonitorElementT.h"

//--------------------------------------------------------------------------------------------
SiStripMonitorDigi::SiStripMonitorDigi(const edm::ParameterSet& iConfig) : dbe_(edm::Service<DaqMonitorBEInterface>().operator->()), conf_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(false), calculate_strip_occupancy(false), reset_each_run(false) {}
SiStripMonitorDigi::~SiStripMonitorDigi() {}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::beginRun(const edm::Run&, const edm::EventSetup&){
  if(reset_each_run){ // reset histograms at beginning of each new run
    for(std::map<uint32_t, ModMEs>::const_iterator idet = DigiMEs.begin(); idet!= DigiMEs.end(); idet++ ){
     ResetME( (idet->second).NumberOfDigis );
     ResetME( (idet->second).ADCsHottestStrip );
     ResetME( (idet->second).ADCsCoolestStrip );
     ResetME( (idet->second).DigiADCs );
     ResetME( (idet->second).StripOccupancy );
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::endRun(const edm::Run&, const edm::EventSetup&){
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file
   show_mechanical_structure_view = conf_.getParameter<bool>("ShowMechanicalStructureView");
   show_readout_view = conf_.getParameter<bool>("ShowReadoutView");
   show_control_view = conf_.getParameter<bool>("ShowControlView");
   select_all_detectors = conf_.getParameter<bool>("SelectAllDetectors");
   calculate_strip_occupancy = conf_.getParameter<bool>("CalculateStripOccupancy");
   reset_each_run = conf_.getParameter<bool>("ResetMEsEachRun");
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowMechanicalStructureView = "<<show_mechanical_structure_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowReadoutView = "<<show_readout_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ShowControlView = "<<show_control_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"SelectAllDetectors = "<<select_all_detectors;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"CalculateStripOccupancy = "<<calculate_strip_occupancy;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi|ConfigParams")<<"ResetMEsEachRun = "<<reset_each_run;

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
//      substructure.getTIBDetectors(activeDets, SelectedDetIds, 1, 1, 1, 1); // this adds rawDetIds to SelectedDetIds
      substructure.getTIBDetectors(activeDets, SelectedDetIds, 2, 0, 0, 0); // this adds rawDetIds to SelectedDetIds
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
//      // create ADCs per strip
//      std::string hid = hidmanager.createHistoId("ADCsPerStrip_detector", *detid_iterator);
//      local_me = dbe_->book2D(hid, hid, 20,-0.5,767.5, 20,-0.5,255.5);
//      ADCsPerStrip.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
      // create Digis per detector - not too useful - maybe can remove later
      hid = hidmanager.createHistoId("NumberOfDigis","det",*detid_iterator);
      local_modmes.NumberOfDigis = dbe_->book1D(hid, hid, 21, -0.5, 20.5); dbe_->tag(local_modmes.NumberOfDigis, *detid_iterator);
      local_modmes.NumberOfDigis->setAxisTitle("number of digis in one detector module");
      // create ADCs per "hottest" strip
      hid = hidmanager.createHistoId("ADCsHottestStrip","det",*detid_iterator);
      local_modmes.ADCsHottestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.); dbe_->tag(local_modmes.ADCsHottestStrip, *detid_iterator);
      local_modmes.ADCsHottestStrip->setAxisTitle("number of ADCs in strip with most of them");
      // create ADCs per "coolest" strip
      hid = hidmanager.createHistoId("ADCsCoolestStrip","det",*detid_iterator);
      local_modmes.ADCsCoolestStrip = dbe_->book1D(hid, hid, 21, -0.5, 50.); dbe_->tag(local_modmes.ADCsCoolestStrip, *detid_iterator);
      local_modmes.ADCsCoolestStrip->setAxisTitle("number of ADCs in strip with less of them");
      // create Digi ADC count distribution
      hid = hidmanager.createHistoId("DigiADCs","det",*detid_iterator);
      local_modmes.DigiADCs = dbe_->book1D(hid, hid, 21, -0.5, 50.); dbe_->tag(local_modmes.DigiADCs, *detid_iterator);
      local_modmes.DigiADCs->setAxisTitle("ADCs");
      // create Strip Occupancy histograms (if flag set to yes in configuration file)
      if(calculate_strip_occupancy){
        hid = hidmanager.createHistoId("StripOccupancy","det",*detid_iterator);
        if(tkmechstruct->nApvPairs(*detid_iterator)==3){ // ask the cabling how many APVs does this detector module have and book histogram accordingly
          local_modmes.StripOccupancy = dbe_->bookProfile(hid, hid, 768, -0.5, 767.5, 10, 0., 1.);
          local_modmes.StripOccupancy->setAxisTitle("Strips [0-767]");
        }else{
          local_modmes.StripOccupancy = dbe_->bookProfile(hid, hid, 512, -0.5, 511.5, 10, 0., 1.);
          local_modmes.StripOccupancy->setAxisTitle("Strips [0-511]");
        }
        local_modmes.StripOccupancy->setAxisTitle("Occupancy",2);
        dbe_->tag(local_modmes.StripOccupancy, *detid_iterator);
      }
      // append to DigiMEs
      DigiMEs.insert( std::make_pair(*detid_iterator, local_modmes));
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
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
      std::vector<uint16_t> non_zero_strips; non_zero_strips.clear(); // keeps list of strips with ADC counts > 0
      // get from DetSetVector the DetSet of digis belonging to one detid - first make sure there exists digis with this id
      edm::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detsetvektor->find(detid); // search  digis of detid
      if(isearch==digi_detsetvektor->end()){
        if(local_modmes.NumberOfDigis != NULL){
          (local_modmes.NumberOfDigis)->Fill(0.,1.); // no digis for this detector module, so fill histogram with 0
        }
        if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip - non_zero_strips has no elements so histogram will be filled with zeros
          FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
        }
        continue; // no digis for this detid => jump to next step of loop
      }
      //digi_detset is a structure, digi_detset.data is a std::vector<SiStripDigi>, digi_detset.id is uint32_t
      edm::DetSet<SiStripDigi> digi_detset = (*digi_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
      //
      if(local_modmes.NumberOfDigis != NULL){ // nr. of digis per detector
        (local_modmes.NumberOfDigis)->Fill(static_cast<float>(digi_detset.data.size()),1.);
      }
      if(digi_detset.data.size()==0){
        if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip - non_zero_strips has no elements so histogram will be filled with zeros
          FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
        }
        continue; // continue with next ME if 0 digis in this detset
      }
      // ADCs
      int largest_adc=(digi_detset.data.begin())->adc();
      int smallest_adc=(digi_detset.data.begin())->adc();
      for(edm::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.data.begin(); digiIter!= digi_detset.data.end(); digiIter++ ){
        int this_adc = digiIter->adc();
        non_zero_strips.push_back(digiIter->strip());
        if(this_adc>largest_adc)  largest_adc  = this_adc; 
        if(local_modmes.DigiADCs != NULL ){ // distribution of ADC counts
          (local_modmes.DigiADCs)->Fill(static_cast<float>(this_adc),1.);
        }
      }
      if(local_modmes.ADCsHottestStrip != NULL){ // nr. of adcs for hottest strip
        (local_modmes.ADCsHottestStrip)->Fill(static_cast<float>(largest_adc),1.);
      }
      if(local_modmes.ADCsCoolestStrip){ // nr. of adcs for coolest strip
        (local_modmes.ADCsCoolestStrip)->Fill(static_cast<float>(smallest_adc),1.);
      }
      if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip
        FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
      }
    }
  }
}

//--------------------------------------------------------------------------------------------
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
     monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ DigiADCs         "<<(idet->second).DigiADCs->getEntries()<<" "<<(idet->second).DigiADCs->getMean()<<" "<<(idet->second).DigiADCs->getRMS()<<std::endl;
    }
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    // save histograms in a file
    dbe_->save(outputFileName);
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::FillStripOccupancy(MonitorElement* StripOccupancy,  std::vector<uint16_t> & non_zero_strips){
  for(uint16_t istrip = 0; istrip<=StripOccupancy->getNbinsX(); ++istrip){
    std::vector<uint16_t>::iterator inonzero = std::find(non_zero_strips.begin(),non_zero_strips.end(),istrip); // is strip part of nonzero list?
    // fill each bin with zeros unless strip in nonzero list
    if( inonzero == non_zero_strips.end()){ // not found in list
        StripOccupancy->Fill(istrip,0.);
    }else{
        StripOccupancy->Fill(istrip,1.);
        non_zero_strips.erase(inonzero); // remove so that do not search this again next time
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::ResetME(MonitorElement* me){
  MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
  if (ob) {
    TH1F * root_ob = dynamic_cast<TH1F *> (ob->operator->());
    if(root_ob)root_ob->Reset();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi);
