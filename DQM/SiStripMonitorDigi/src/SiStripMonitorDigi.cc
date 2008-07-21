// -*- c++ -*-
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc
 */
// Original Author:  Dorian Kcira
//         Created:  Sat Feb  4 20:49:10 CET 2006
// $Id: SiStripMonitorDigi.cc,v 1.25 2008/04/28 19:46:32 dutta Exp $
#include<fstream>
#include "TNamed.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TMath.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"


//--------------------------------------------------------------------------------------------
SiStripMonitorDigi::SiStripMonitorDigi(const edm::ParameterSet& iConfig) : dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(true), calculate_strip_occupancy(false), reset_each_run(false), m_cacheID_(0), folder_organizer() 
{
  firstEvent = -1;
  eventNb = 0;

  NDigi = new int*[4];
  NDigi[0] = new int[4];
  NDigi[1] = new int[6];
  NDigi[2] = new int[6];
  NDigi[3] = new int[18];

  ADCHottest = new int*[4];
  ADCHottest[0] = new int[4];
  ADCHottest[1] = new int[6];
  ADCHottest[2] = new int[6];
  ADCHottest[3] = new int[18];

  ADCCoolest = new int*[4];
  ADCCoolest[0] = new int[4];
  ADCCoolest[1] = new int[6];
  ADCCoolest[2] = new int[6];
  ADCCoolest[3] = new int[18];


  //get on/off option for every cluster from cfi
  edm::ParameterSet ParametersNumberOfDigis =  conf_.getParameter<edm::ParameterSet>("TH1NumberOfDigis");
  layerswitchnumdigison = ParametersNumberOfDigis.getParameter<bool>("layerswitchon");
  moduleswitchnumdigison = ParametersNumberOfDigis.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersADCsHottestStrip =  conf_.getParameter<edm::ParameterSet>("TH1ADCsHottestStrip");
  layerswitchadchotteston = ParametersADCsHottestStrip.getParameter<bool>("layerswitchon");
  moduleswitchadchotteston = ParametersADCsHottestStrip.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersADCsCoolestStrip =  conf_.getParameter<edm::ParameterSet>("TH1ADCsCoolestStrip");
  layerswitchadccooleston = ParametersADCsCoolestStrip.getParameter<bool>("layerswitchon");
  moduleswitchadccooleston = ParametersADCsCoolestStrip.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersDigiADCs =  conf_.getParameter<edm::ParameterSet>("TH1DigiADCs");
  layerswitchdigiadcson = ParametersDigiADCs.getParameter<bool>("layerswitchon");
  moduleswitchdigiadcson = ParametersDigiADCs.getParameter<bool>("moduleswitchon");
  

  edm::ParameterSet ParametersDetsOn =  conf_.getParameter<edm::ParameterSet>("detectorson");
  tibon = ParametersDetsOn.getParameter<bool>("tibon");
  tidon = ParametersDetsOn.getParameter<bool>("tidon");
  tobon = ParametersDetsOn.getParameter<bool>("tobon");
  tecon = ParametersDetsOn.getParameter<bool>("tecon");

}



SiStripMonitorDigi::~SiStripMonitorDigi() { }

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::beginRun(const edm::Run& run, const edm::EventSetup& es){

  if (show_mechanical_structure_view) {
    unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    if (m_cacheID_ != cacheID) {
      m_cacheID_ = cacheID;       
      edm::LogInfo("SiStripMonitorDigi") <<"SiStripMonitorDigi::beginRun: " 
					 << " Creating MEs for new Cabling ";     
      createMEs(es);
    } 
  } else if (reset_each_run) {
    edm::LogInfo("SiStripMonitorDigi") <<"SiStripMonitorDigi::beginRun: " 
				       << " Resetting MEs ";        
    for (std::map<uint32_t, ModMEs >::const_iterator idet = DigiMEs.begin() ; idet!=DigiMEs.end() ; idet++) {
      ResetModuleMEs(idet->first);
    }
  }


  es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );
  bookLayer();

}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::endRun(const edm::Run&, const edm::EventSetup&){
}



//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::beginJob(const edm::EventSetup& es){
}


//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::createMEs(const edm::EventSetup& es){

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

      if(tibon) substructure.getTIBDetectors(activeDets, SelectedDetIds, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds
      if(tobon) substructure.getTOBDetectors(activeDets, SelectedDetIds, 0, 0, 0);    // this adds rawDetIds to SelectedDetIds
      if(tidon) substructure.getTIDDetectors(activeDets, SelectedDetIds, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds
      if(tecon) substructure.getTECDetectors(activeDets, SelectedDetIds, 0, 0, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds

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

      if (reset_each_run) ResetModuleMEs(*detid_iterator);

      //      // create ADCs per strip
      //      std::string hid = hidmanager.createHistoId("ADCsPerStrip_detector", *detid_iterator);
      //      local_me = dqmStore_->book2D(hid, hid, 20,-0.5,767.5, 20,-0.5,255.5);
      //      ADCsPerStrip.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );

      if(moduleswitchnumdigison) {
	// create Digis per detector - not too useful - maybe can remove later
	hid = hidmanager.createHistoId("NumberOfDigis","det",*detid_iterator);
	local_modmes.NumberOfDigis = dqmStore_->book1D(hid, hid, 21, -0.5, 20.5); dqmStore_->tag(local_modmes.NumberOfDigis, *detid_iterator);
	local_modmes.NumberOfDigis->setAxisTitle("number of digis in one detector module");
      }

      if(moduleswitchadchotteston) {
	// create ADCs per "hottest" strip
	hid = hidmanager.createHistoId("ADCsHottestStrip","det",*detid_iterator);
	local_modmes.ADCsHottestStrip = dqmStore_->book1D(hid, hid, 21, -0.5, 50.); dqmStore_->tag(local_modmes.ADCsHottestStrip, *detid_iterator);
	local_modmes.ADCsHottestStrip->setAxisTitle("number of ADCs in strip with most of them");
      }

      if(moduleswitchadccooleston) {
	// create ADCs per "coolest" strip
	hid = hidmanager.createHistoId("ADCsCoolestStrip","det",*detid_iterator);
	local_modmes.ADCsCoolestStrip = dqmStore_->book1D(hid, hid, 21, -0.5, 50.); dqmStore_->tag(local_modmes.ADCsCoolestStrip, *detid_iterator);
	local_modmes.ADCsCoolestStrip->setAxisTitle("number of ADCs in strip with less of them");
      }

      if(moduleswitchdigiadcson) {
	// create Digi ADC count distribution
	hid = hidmanager.createHistoId("DigiADCs","det",*detid_iterator);
	local_modmes.DigiADCs = dqmStore_->book1D(hid, hid, 21, -0.5, 50.); dqmStore_->tag(local_modmes.DigiADCs, *detid_iterator);
	local_modmes.DigiADCs->setAxisTitle("ADCs");
      }

      //       // create Strip Occupancy histograms (if flag set to yes in configuration file)
      //       if(calculate_strip_occupancy){
      //         hid = hidmanager.createHistoId("StripOccupancy","det",*detid_iterator);
      //         if(tkmechstruct->nApvPairs(*detid_iterator)==3){ // ask the cabling how many APVs does this detector module have and book histogram accordingly
      //           local_modmes.StripOccupancy = dqmStore_->bookProfile(hid, hid, 768, -0.5, 767.5, 10, 0., 1.);
      //           local_modmes.StripOccupancy->setAxisTitle("Strips [0-767]");
      //         }else{
      //           local_modmes.StripOccupancy = dqmStore_->bookProfile(hid, hid, 512, -0.5, 511.5, 10, 0., 1.);
      //           local_modmes.StripOccupancy->setAxisTitle("Strips [0-511]");
      //         }
      //         local_modmes.StripOccupancy->setAxisTitle("Occupancy",2);
      //         dqmStore_->tag(local_modmes.StripOccupancy, *detid_iterator);
      //       }


      // append to DigiMEs
      DigiMEs.insert( std::make_pair(*detid_iterator, local_modmes));
    }
  }
}



//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){


  using namespace edm;

  runNb   = iEvent.id().run();
  //   eventNb = iEvent.id().event();
  eventNb++;


  //initialize to zero all arrays that will contain
  //#digis, #ADCs for hottest and coolest strip per event
  for(int i=0;i<4;i++) 
    {
      switch(i) {
      case 0:
	{
	  for(int j=0;j<4;j++) {NDigi[i][j]=0;ADCHottest[i][j]=0;ADCCoolest[i][j]=10000;}//TIB
	  break;
	}
      case 1:
	{
	  for(int j=0;j<6;j++) {NDigi[i][j]=0;ADCHottest[i][j]=0;ADCCoolest[i][j]=10000;}//TIB
	  break;
	}
      case 2:
	{
	  for(int j=0;j<6;j++) {NDigi[i][j]=0;ADCHottest[i][j]=0;ADCCoolest[i][j]=10000;}//TIB
	  break;
	}
      case 3:
	{
	  for(int j=0;j<18;j++) {NDigi[i][j]=0;ADCHottest[i][j]=0;ADCCoolest[i][j]=10000;}//TIB
	  break;
	}
      }
    }


  // get all digi collections
  //edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
  Parameters::iterator itDigiProducersList = DigiProducersList.begin();

  for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {

    std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
    std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
    iEvent.getByLabel(digiProducer,digiLabel,digi_detsetvektor);
    
    if (!digi_detsetvektor.isValid()) continue; 
    
    // loop over all MEs
    for (std::map<uint32_t, ModMEs >::const_iterator iterMEs = DigiMEs.begin() ; iterMEs!=DigiMEs.end() ; iterMEs++) {
    
      // get detid and type of ME
      uint32_t detid = iterMEs->first; 
      ModMEs local_modmes = iterMEs->second; 

      // keeps list of strips with ADC counts > 0      
      std::vector<uint16_t> non_zero_strips; 
      non_zero_strips.clear(); 

      // get from DetSetVector the DetSet of digis belonging to one detid 
      // first make sure there exists digis with this id
      // edmNew::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detsetvektor->find(detid); 
      // search  digis of detid
      // edm::New does not work because collection is not this type (why ?????)

      // search  digis of detid
      edm::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detsetvektor->find(detid); 

      if(isearch==digi_detsetvektor->end()){

	if(moduleswitchnumdigison) {
	  if(local_modmes.NumberOfDigis != NULL){
	    // no digis for this detector module, so fill histogram with 0
	    (local_modmes.NumberOfDigis)->Fill(0.,1.); 
	  }
	}

        if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip - non_zero_strips has no elements so histogram will be filled with zeros
          //FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
        }

        continue; // no digis for this detid => jump to next step of loop
      }//end of if "isearch == digi ..."

      //digi_detset is a structure
      //digi_detset.data is a std::vector<SiStripDigi>
      //digi_detset.id is uint32_t
      edm::DetSet<SiStripDigi> digi_detset = (*digi_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'

      if(moduleswitchnumdigison) {
	if(local_modmes.NumberOfDigis != NULL){ // nr. of digis per detector
	  //(local_modmes.NumberOfDigis)->Fill(static_cast<float>(digi_detset.data.size()),1.);
	  (local_modmes.NumberOfDigis)->Fill(static_cast<float>(digi_detset.size()),1.);
	}
      }

      //if(digi_detset.data.size()==0){
      if(digi_detset.size()==0){
        if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip - non_zero_strips has no elements so histogram will be filled with zeros
	  //FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
        }

        continue; // continue with next ME if 0 digis in this detset
      }


      // ADCs
      int largest_adc=(digi_detset.data.begin())->adc();
      int smallest_adc=(digi_detset.data.begin())->adc();

      for(edm::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.data.begin(); 
	  digiIter!= digi_detset.data.end(); digiIter++ ){

	//       int largest_adc=(digi_detset.begin())->adc();
	//       int smallest_adc=(digi_detset.begin())->adc();
	//       for(edmNew::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.begin(); digiIter!= digi_detset.end(); digiIter++ ){

        int this_adc = digiIter->adc();
        non_zero_strips.push_back(digiIter->strip());

        if(this_adc>largest_adc)  largest_adc  = this_adc; 

	if(moduleswitchdigiadcson) {
	  if(local_modmes.DigiADCs != NULL ){ // distribution of ADC counts
	    (local_modmes.DigiADCs)->Fill(static_cast<float>(this_adc),1.);
	  }
	}

      }//end of loop over digis in this det

      if(moduleswitchadchotteston) {
	if(local_modmes.ADCsHottestStrip != NULL){ // nr. of adcs for hottest strip
	  (local_modmes.ADCsHottestStrip)->Fill(static_cast<float>(largest_adc),1.);
	}
      }

      if(moduleswitchadccooleston) {
	if(local_modmes.ADCsCoolestStrip){ // nr. of adcs for coolest strip
	  (local_modmes.ADCsCoolestStrip)->Fill(static_cast<float>(smallest_adc),1.);
	}
      }

      if(calculate_strip_occupancy && local_modmes.StripOccupancy){ // nr. of adcs for coolest strip
        //FillStripOccupancy(local_modmes.StripOccupancy, non_zero_strips);
      }

    }//end of loop over MEs

  }//end of loop over digi producers (ZeroSuppressed, VirginRaw, ProcessedRaw, ScopeMode)

  AllDigis(iSetup);

}//end of method analyze



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

      if(moduleswitchnumdigison) {     
	monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ NumberOfDigis "<<(idet->second).NumberOfDigis->getEntries()<<" "<<(idet->second).NumberOfDigis->getMean()<<" "<<(idet->second).NumberOfDigis->getRMS()<<std::endl;
      }

      if(moduleswitchadchotteston) {     
	monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ ADCsHottestStrip "<<(idet->second).ADCsHottestStrip->getEntries()<<" "<<(idet->second).ADCsHottestStrip->getMean()<<" "<<(idet->second).ADCsHottestStrip->getRMS()<<std::endl;
      }

      if(moduleswitchadccooleston) {     
	monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ ADCsCoolestStrip "<<(idet->second).ADCsCoolestStrip->getEntries()<<" "<<(idet->second).ADCsCoolestStrip->getMean()<<" "<<(idet->second).ADCsCoolestStrip->getRMS()<<std::endl;
      }

      if(moduleswitchdigiadcson) {     
	monitor_summary<<"SiStripTkDQM|SiStripMonitorDigi"<<"              +++ DigiADCs         "<<(idet->second).DigiADCs->getEntries()<<" "<<(idet->second).DigiADCs->getMean()<<" "<<(idet->second).DigiADCs->getRMS()<<std::endl;
      }
    
    }//end of loop over MEs

    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    
  }//end of if

  // save histograms in a file
  dqmStore_->save(outputFileName);
  
}//end of method



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
void SiStripMonitorDigi::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = DigiMEs.find(idet);
  ModMEs mod_me = pos->second;

  if(moduleswitchnumdigison) mod_me.NumberOfDigis->Reset();
  if(moduleswitchadchotteston) mod_me.ADCsHottestStrip->Reset();
  if(moduleswitchadccooleston) mod_me.ADCsCoolestStrip->Reset();
  if(moduleswitchdigiadcson) mod_me.DigiADCs->Reset();
  //mod_me.StripOccupancy->Reset();

}
























































//------------------------------------------------------------------------  
void SiStripMonitorDigi::bookLayer() 
{
  
  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);

  std::vector<uint32_t> LayerSelectedDetIds;
  if(select_all_detectors){
    // select all detectors if appropriate flag is set,  for example for the mtcc
    LayerSelectedDetIds = vdetId_;
  }else{
    // use SiStripSubStructure for selecting certain regions
    SiStripSubStructure substructure;

    if(tibon) substructure.getTIBDetectors(vdetId_, LayerSelectedDetIds, 0, 0, 0, 0); // this adds rawDetIds to LayerSelectedDetIds
    if(tobon) substructure.getTOBDetectors(vdetId_, LayerSelectedDetIds, 0, 0, 0);    // this adds rawDetIds to LayerSelectedDetIds
    if(tidon) substructure.getTIDDetectors(vdetId_, LayerSelectedDetIds, 0, 0, 0, 0); // this adds rawDetIds to LayerSelectedDetIds
    if(tecon) substructure.getTECDetectors(vdetId_, LayerSelectedDetIds, 0, 0, 0, 0, 0, 0); // this adds rawDetIds to LayerSelectedDetIds
  }


  // remove any eventual zero elements - there should be none, but just in case
  for(std::vector<uint32_t>::iterator idets = LayerSelectedDetIds.begin(); idets != LayerSelectedDetIds.end(); idets++){
    if(*idets == 0) LayerSelectedDetIds.erase(idets);
  }
  

  //Histos for each detector, layer and module
  for (std::vector<uint32_t>::const_iterator detid_iter=LayerSelectedDetIds.begin();detid_iter!=LayerSelectedDetIds.end();detid_iter++){  //loop on all the active detid
    uint32_t detid = *detid_iter;

    if (detid < 1){
      edm::LogError("SiStripMonitorDigi")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
      continue;
    }
    if (DetectedLayers.find(folder_organizer.GetSubDetAndLayer(detid)) == DetectedLayers.end()){
      DetectedLayers[folder_organizer.GetSubDetAndLayer(detid)]=true;
    }    

    // book Layer plots      
    std::string flagtempo = "";
    folder_organizer.setLayerFolder(*detid_iter,folder_organizer.GetSubDetAndLayer(*detid_iter).second); 
    bookTrendMEs("layer",folder_organizer.GetSubDetAndLayer(*detid_iter).second,*detid_iter,flagtempo);
    //    }
  
  }//end loop on detector

}
  






void SiStripMonitorDigi::bookTrendMEs(TString name,int32_t layer,uint32_t id,std::string flag)//Histograms and Trends at LAYER LEVEL
{
  char rest[1024];
  int subdetid = ((id>>25)&0x7);
  if(       subdetid==3 ){
    // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(id);
    sprintf(rest,"TIB__layer__%d",tib1.layer());
  }else if( subdetid==4){
    // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(id);
    sprintf(rest,"TID__side__%d__wheel__%d",tid1.side(),tid1.wheel());
  }else if( subdetid==5){
    // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(id);
    sprintf(rest,"TOB__layer__%d",tob1.layer());
  }else if( subdetid==6){
    // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(id);
    sprintf(rest,"TEC__side__%d__wheel__%d",tec1.side(),tec1.wheel());
  }else{
    // ---------------------------  ???  --------------------------- //
    edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<std::endl;
    return;
  }

  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoLayer("",name.Data(),rest,flag);
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(hid));
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 


    //Number of Digis
    //MonitorElement* LayerNumberOfDigis;

    if(layerswitchnumdigison) {
      theModMEs.LayerNumberOfDigis=bookME1D("TH1NumberOfDigis", hidmanager.createHistoLayer("Summary_numDigis",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerNumberOfDigis,layer); 
      theModMEs.LayerNumberOfDigisTrend=bookMETrend("TH1NumberOfDigis", hidmanager.createHistoLayer("Trend_numDigis",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerNumberOfDigisTrend,layer); 
    }

    //Number of Digis for hottest strip
    //MonitorElement* ADCsHottestStrip;

    if(layerswitchadchotteston) {
      theModMEs.LayerADCsHottestStrip=bookME1D("TH1ADCsHottestStrip", hidmanager.createHistoLayer("Summary_ADCsHottestStrip",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerADCsHottestStrip,layer); 
      theModMEs.LayerADCsHottestStripTrend=bookMETrend("TH1ADCsHottestStrip", hidmanager.createHistoLayer("Trend_ADCsHottestStrip",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerADCsHottestStripTrend,layer); 
    }

    //Number of Digis for coolest strip
    //MonitorElement* ADCsCoolestStrip;

    if(layerswitchadccooleston) {
      theModMEs.LayerADCsCoolestStrip=bookME1D("TH1ADCsCoolestStrip", hidmanager.createHistoLayer("Summary_ADCsCoolestStrip",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerADCsCoolestStrip,layer); 
      theModMEs.LayerADCsCoolestStripTrend=bookMETrend("TH1ADCsCoolestStrip", hidmanager.createHistoLayer("Trend_ADCsCoolestStrip",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerADCsCoolestStripTrend,layer); 
    }

    //ADCs per Digi
    //MonitorElement* DigiADCs ;

    if(layerswitchdigiadcson) {
      theModMEs.LayerDigiADCs=bookME1D("TH1DigiADCs", hidmanager.createHistoLayer("Summary_DigiADCs",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerDigiADCs,layer); 
      theModMEs.LayerDigiADCsTrend=bookMETrend("TH1DigiADCs", hidmanager.createHistoLayer("Trend_DigiADCs",name.Data(),rest,flag).c_str()); 
      dqmStore_->tag(theModMEs.LayerDigiADCsTrend,layer); 
    }

    //     //SiStrip Occupancy
    //     //MonitorElement* StripOccupancy;

    //     theModMEs.LayerStripOccupancy=bookME1D("TH1StripOccupancy", hidmanager.createHistoLayer("Summary_StripOccupancy",name.Data(),rest,flag).c_str()); 
    //     dqmStore_->tag(theModMEs.LayerStripOccupancy,layer); 
    //     theModMEs.LayerStripOccupancyTrend=bookMETrend("TH1StripOccupancy", hidmanager.createHistoLayer("Trend_StripOccupancy",name.Data(),rest,flag).c_str()); 
    //     dqmStore_->tag(theModMEs.LayerStripOccupancyTrend,layer); 


    //bookeeping
    ModMEsMap[hid]=theModMEs;
  }

}






MonitorElement* SiStripMonitorDigi::bookMETrend(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  edm::ParameterSet ParametersTrend =  conf_.getParameter<edm::ParameterSet>("Trending");
  MonitorElement* me = dqmStore_->bookProfile(HistoName,HistoName,
					      ParametersTrend.getParameter<int32_t>("Nbins"),
					      // 					      0,
					      ParametersTrend.getParameter<double>("xmin"),
					      ParametersTrend.getParameter<double>("xmax"),
					      // 					      ParametersTrend.getParameter<int32_t>("Nbins"),
					      100, //that parameter should not be there !?
					      ParametersTrend.getParameter<double>("ymin"),
					      ParametersTrend.getParameter<double>("ymax"),
					      "" );
  if(!me) return me;
  char buffer[256];
  sprintf(buffer,"EventId/%d",ParametersTrend.getParameter<int32_t>("Steps"));
  me->setAxisTitle(std::string(buffer),1);
  return me;
}

//------------------------------------------------------------------------------------------






MonitorElement* SiStripMonitorDigi::bookME1D(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dqmStore_->book1D(HistoName,HistoName,
			   Parameters.getParameter<int32_t>("Nbinx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax")
			   );
}




void SiStripMonitorDigi::fillTrendMEs(int subdetid, int subsubdetid, std::string name)
{ 

  //   std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(name));
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);

  if(iModME!=ModMEsMap.end()){
    
    if(layerswitchnumdigison) {
      fillME(iModME->second.LayerNumberOfDigis  ,NDigi[(subdetid)][subsubdetid]);
      fillTrend(iModME->second.LayerNumberOfDigisTrend,NDigi[(subdetid)][subsubdetid]);
    }
    if(layerswitchadchotteston) {
      fillME(iModME->second.LayerADCsHottestStrip,ADCHottest[(subdetid)][subsubdetid]);
      fillTrend(iModME->second.LayerADCsHottestStripTrend,ADCHottest[(subdetid)][subsubdetid]);
    }
    if(layerswitchadccooleston) {
      fillME(iModME->second.LayerADCsCoolestStrip ,ADCCoolest[(subdetid)][subsubdetid]);
      fillTrend(iModME->second.LayerADCsCoolestStripTrend,ADCCoolest[(subdetid)][subsubdetid]);
    }

  }

}//end of method fillTrendMEs





void SiStripMonitorDigi::fillDigiADCsMEs(int value, std::string name)
{ 
  
  //   std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(name));
  std::map<TString, ModMEs>::iterator iModME = ModMEsMap.find(name);

  if(iModME!=ModMEsMap.end()){

    if(layerswitchdigiadcson) {
      fillME(iModME->second.LayerDigiADCs , value);
      fillTrend(iModME->second.LayerDigiADCsTrend, value);
    }

  }

}//end of method fillDigiADCsMEs




//--------------------------------------------------------------------------------
void SiStripMonitorDigi::fillTrend(MonitorElement* me ,float value)
{
  if(!me) return;
  //check the origin and check options
  int option = conf_.getParameter<edm::ParameterSet>("Trending").getParameter<int32_t>("UpdateMode");
  if(firstEvent==-1) firstEvent = eventNb;
  int CurrentStep = atoi(me->getAxisTitle(1).c_str()+8);
  int firstEventUsed = firstEvent;
  int presentOverflow = (int)me->getBinEntries(me->getNbinsX()+1);
  if(option==2) firstEventUsed += CurrentStep * int(me->getBinEntries(me->getNbinsX()+1));
  else if(option==3) firstEventUsed += CurrentStep * int(me->getBinEntries(me->getNbinsX()+1)) * me->getNbinsX();
  //fill
  me->Fill((eventNb-firstEventUsed)/CurrentStep,value);

  if(eventNb-firstEvent<1) return;
  // check if we reached the end
  if(presentOverflow == me->getBinEntries(me->getNbinsX()+1)) return;
  switch(option) {
  case 1:
    {
      // mode 1: rebin and change X scale
      int NbinsX = me->getNbinsX();
      float entries = 0.;
      float content = 0.;
      float error = 0.;
      int bin = 1;
      int totEntries = int(me->getEntries());
      for(;bin<=NbinsX/2;++bin) {
	content = (me->getBinContent(2*bin-1) + me->getBinContent(2*bin))/2.; 
	error   = pow((me->getBinError(2*bin-1)*me->getBinError(2*bin-1)) + (me->getBinError(2*bin)*me->getBinError(2*bin)),0.5)/2.; 
	entries = me->getBinEntries(2*bin-1) + me->getBinEntries(2*bin);
	me->setBinContent(bin,content*entries);
	me->setBinError(bin,error);
	me->setBinEntries(bin,entries);
      }
      for(;bin<=NbinsX+1;++bin) {
	me->setBinContent(bin,0);
	me->setBinError(bin,0);
	me->setBinEntries(bin,0); 
      }
      me->setEntries(totEntries);
      char buffer[256];
      sprintf(buffer,"EventId/%d",CurrentStep*2);
      me->setAxisTitle(std::string(buffer),1);
      break;
    }
  case 2:
    {
      // mode 2: slide
      int bin=1;
      int NbinsX = me->getNbinsX();
      for(;bin<=NbinsX;++bin) {
	me->setBinContent(bin,me->getBinContent(bin+1)*me->getBinEntries(bin+1));
	me->setBinError(bin,me->getBinError(bin+1));
	me->setBinEntries(bin,me->getBinEntries(bin+1));
      }
      break;
    }
  case 3:
    {
      // mode 3: reset
      int NbinsX = me->getNbinsX();
      for(int bin=0;bin<=NbinsX;++bin) {
	me->setBinContent(bin,0);
	me->setBinError(bin,0);
	me->setBinEntries(bin,0); 
      }
      break;
    }
  }
}




bool SiStripMonitorDigi::AllDigis( const edm::EventSetup& es)
{

  using namespace std;

  //Loop on Dets
  // edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=digi_detsetvektor->begin();
  //   for ( edmNew::DetSetVector<SiStripDigi>::const_iterator DSViter=digi_detsetvektor->begin(); DSViter!=digi_detsetvektor->end();DSViter++){
  for ( edm::DetSetVector<SiStripDigi>::const_iterator DSViter=digi_detsetvektor->begin(); DSViter!=digi_detsetvektor->end();DSViter++){
    
    //     uint32_t detid=DSViter->id();
    uint32_t detid=DSViter->id;

    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
    //Loop on Digis
    LogDebug("SiStripMonitorDigi") << "on detid "<< detid << " N Digi= " << DSViter->size();
    //     edmNew::DetSet<SiStripDigi>::const_iterator DigiIter = DSViter->begin();
    
    edm::DetSet<SiStripDigi>::const_iterator DigiIter = DSViter->begin();
    for(; DigiIter!=DSViter->end(); DigiIter++) {

      std::string name;
      std::string flag = "";
      char rest[1024];

      int this_adc = DigiIter->adc();

      int subdetid = ((detid>>25)&0x7);
      //This is the id of the layer or of the wheel
      int subsubdetid = 14;
      if(       subdetid==3 ){
	// ---------------------------  TIB  --------------------------- //
	TIBDetId tib1 = TIBDetId(detid);
	sprintf(rest,"TIB__layer__%d",tib1.layer());
	subsubdetid = tib1.layer()-1;
	if(this_adc > ADCHottest[(subdetid-3)][(subsubdetid)])  ADCHottest[(subdetid-3)][(subsubdetid)]= this_adc;
	if(this_adc < ADCCoolest[(subdetid-3)][(subsubdetid)])  ADCCoolest[(subdetid-3)][(subsubdetid)]= this_adc;
      }else if( subdetid==4){
	// ---------------------------  TID  --------------------------- //
	TIDDetId tid1 = TIDDetId(detid);
	sprintf(rest,"TID__side__%d__wheel__%d",tid1.side(),tid1.wheel());
	subsubdetid = (tid1.wheel()-1)+3*(tid1.side()-1);
	if(this_adc > ADCHottest[(subdetid-3)][(subsubdetid)])  ADCHottest[(subdetid-3)][(subsubdetid)]= this_adc;
	if(this_adc < ADCCoolest[(subdetid-3)][(subsubdetid)])  ADCCoolest[(subdetid-3)][(subsubdetid)]= this_adc;
      }else if( subdetid==5){
	// ---------------------------  TOB  --------------------------- //
	TOBDetId tob1 = TOBDetId(detid);
	sprintf(rest,"TOB__layer__%d",tob1.layer());
	subsubdetid = tob1.layer()-1;
	if(this_adc > ADCHottest[(subdetid-3)][(subsubdetid)])  ADCHottest[(subdetid-3)][(subsubdetid)]= this_adc;
	if(this_adc < ADCCoolest[(subdetid-3)][(subsubdetid)])  ADCCoolest[(subdetid-3)][(subsubdetid)]= this_adc;
      }else if( subdetid==6){
	// ---------------------------  TEC  --------------------------- //
	TECDetId tec1 = TECDetId(detid);
	sprintf(rest,"TEC__side__%d__wheel__%d",tec1.side(),tec1.wheel());
	subsubdetid = (tec1.wheel()-1)+9*(tec1.side()-1);
 	if(this_adc > ADCHottest[(subdetid-3)][(subsubdetid)])  ADCHottest[(subdetid-3)][(subsubdetid)]= this_adc;
	if(this_adc < ADCCoolest[(subdetid-3)][(subsubdetid)])  ADCCoolest[(subdetid-3)][(subsubdetid)]= this_adc;
      }else{
	// ---------------------------  ???  --------------------------- //
	edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<std::endl;
	return 0;
      }
      
      //std::cout<<" subsubdetid "<<subsubdetid<<endl;
      if(subsubdetid < 14) NDigi[(subdetid-3)][(subsubdetid)]++;

      SiStripHistoId hidmanager;
      name= hidmanager.createHistoLayer("","layer",rest,flag);
      fillDigiADCsMEs(this_adc, name);


    }//end of loop over digis
    

  }//end of loop on detsetvectors


  //Here we should make sure everything is ok

  std::string name;
  char rest[1024];
  std::string flag = "";

  for(int i=0;i<4;i++) 
    {
      switch(i) {
      case 0:
	{
	  for(int j=0;j<4;j++) {
	    //std::cout<<" NDigi[i][j] "<<NDigi[i][j]<<std::endl;
	    //std::cout<<" ADCHottest[i][j] "<<ADCHottest[i][j]<<std::endl;
	    //std::cout<<" ADCCoolest[i][j] "<<ADCCoolest[i][j]<<std::endl;
	    sprintf(rest,"TIB__layer__%d",(j+1));
	    SiStripHistoId hidmanager;
	    name= hidmanager.createHistoLayer("","layer",rest,flag);
	    //std::cout << name.c_str() << std::endl;  
	    fillTrendMEs(i,j,name);
	  }
	  break;
	}
      case 1:
	{
	  for(int j=0;j<6;j++) {
	    //std::cout<<" NDigi[i][j] "<<NDigi[i][j]<<std::endl;
	    //std::cout<<" ADCHottest[i][j] "<<ADCHottest[i][j]<<std::endl;
	    //std::cout<<" ADCCoolest[i][j] "<<ADCCoolest[i][j]<<std::endl;
	    sprintf(rest,"TID__side__%d__wheel__%d",(j/3)+1,(j-((j/3)*3))+1);
	    SiStripHistoId hidmanager;
	    name= hidmanager.createHistoLayer("","layer",rest,flag);
	    //std::cout << name.c_str() << std::endl;  
	    fillTrendMEs(i,j,name);
	  }
	  break;
	}
      case 2:
	{
	  for(int j=0;j<6;j++) {
	    //std::cout<<" NDigi[i][j] "<<NDigi[i][j]<<std::endl;
	    //std::cout<<" ADCHottest[i][j] "<<ADCHottest[i][j]<<std::endl;
	    //std::cout<<" ADCCoolest[i][j] "<<ADCCoolest[i][j]<<std::endl;
	    sprintf(rest,"TOB__layer__%d",(j+1));
	    SiStripHistoId hidmanager;
	    name= hidmanager.createHistoLayer("","layer",rest,flag);
	    //std::cout << name.c_str() << std::endl;  
	    fillTrendMEs(i,j,name);
	  }
	  break;
	}
      case 3:
	{
	  for(int j=0;j<18;j++) {
	    //std::cout<<" NDigi[i][j] "<<NDigi[i][j]<<std::endl;
	    //std::cout<<" ADCHottest[i][j] "<<ADCHottest[i][j]<<std::endl;
	    //std::cout<<" ADCCoolest[i][j] "<<ADCCoolest[i][j]<<std::endl;
	    sprintf(rest,"TEC__side__%d__wheel__%d",(j/9)+1,(j-((j/9)*9))+1);
	    SiStripHistoId hidmanager;
	    name= hidmanager.createHistoLayer("","layer",rest,flag);
	    //std::cout << name.c_str() << std::endl;  
	    fillTrendMEs(i,j,name);
	  }
	  break;
	}
      }
    }

 
}





//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi);
