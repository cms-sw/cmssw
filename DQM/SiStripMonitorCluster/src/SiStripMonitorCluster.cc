// -*- C++ -*-
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc
 */
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
// $Id: SiStripMonitorCluster.cc,v 1.41 2008/05/12 15:24:04 dutta Exp $
#include <vector>
#include <numeric>
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "TMath.h"
#include <iostream>

//--------------------------------------------------------------------------------------------
SiStripMonitorCluster::SiStripMonitorCluster(const edm::ParameterSet& iConfig) : dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(false), reset_each_run(false), m_cacheID_(0)
{

  firstEvent = -1;
  eventNb = 0;


  //get on/off option for every cluster from cfi
  edm::ParameterSet ParametersnClusters =  conf_.getParameter<edm::ParameterSet>("TH1nClusters");
  layerswitchncluson = ParametersnClusters.getParameter<bool>("layerswitchon");
  moduleswitchncluson = ParametersnClusters.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterCharge =  conf_.getParameter<edm::ParameterSet>("TH1ClusterCharge");
  layerswitchcluschargeon = ParametersClusterCharge.getParameter<bool>("layerswitchon");
  moduleswitchcluschargeon = ParametersClusterCharge.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterStoN =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
  layerswitchclusstonon = ParametersClusterStoN.getParameter<bool>("layerswitchon");
  moduleswitchclusstonon = ParametersClusterStoN.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterPos =  conf_.getParameter<edm::ParameterSet>("TH1ClusterPos");
  layerswitchclusposon = ParametersClusterPos.getParameter<bool>("layerswitchon");
  moduleswitchclusposon = ParametersClusterPos.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterNoise =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
  layerswitchclusnoiseon = ParametersClusterNoise.getParameter<bool>("layerswitchon");
  moduleswitchclusnoiseon = ParametersClusterNoise.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterWidth =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
  layerswitchcluswidthon = ParametersClusterWidth.getParameter<bool>("layerswitchon");
  moduleswitchcluswidthon = ParametersClusterWidth.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersModuleLocalOccupancy =  conf_.getParameter<edm::ParameterSet>("TH1ModuleLocalOccupancy");
  layerswitchlocaloccupancy = ParametersModuleLocalOccupancy.getParameter<bool>("layerswitchon");
  moduleswitchlocaloccupancy = ParametersModuleLocalOccupancy.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersNrOfClusterizedStrips =  conf_.getParameter<edm::ParameterSet>("TH1NrOfClusterizedStrips");
  layerswitchnrclusterizedstrip = ParametersNrOfClusterizedStrips.getParameter<bool>("layerswitchon");
  moduleswitchnrclusterizedstrip = ParametersNrOfClusterizedStrips.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersDetsOn =  conf_.getParameter<edm::ParameterSet>("detectorson");
  tibon = ParametersDetsOn.getParameter<bool>("tibon");
  tidon = ParametersDetsOn.getParameter<bool>("tidon");
  tobon = ParametersDetsOn.getParameter<bool>("tobon");
  tecon = ParametersDetsOn.getParameter<bool>("tecon");

  createTrendMEs = conf_.getParameter<bool>("CreateTrendMEs");

} 


SiStripMonitorCluster::~SiStripMonitorCluster() { }

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::beginRun(const edm::Run& run, const edm::EventSetup& es){

  if (show_mechanical_structure_view) {
    unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    if (m_cacheID_ != cacheID) {
      m_cacheID_ = cacheID;       
     edm::LogInfo("SiStripMonitorCluster") <<"SiStripMonitorCluster::beginRun: " 
					    << " Creating MEs for new Cabling ";     

      createMEs(es);
    } 
  } else if (reset_each_run) {
    edm::LogInfo("SiStripMonitorCluster") <<"SiStripMonitorCluster::beginRun: " 
					  << " Resetting MEs ";        
    for (std::map<uint32_t, ModMEs >::const_iterator idet = ModuleMEMap.begin() ; idet!=ModuleMEMap.end() ; idet++) {
      ResetModuleMEs(idet->first);
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::createMEs(const edm::EventSetup& es){
  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
    
    // get list of active detectors from SiStripDetCabling 
    std::vector<uint32_t> activeDets;
    SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);
    
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
    SiStripFolderOrganizer folder_organizer;
    folder_organizer.setSiStripFolder();

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
    for(std::vector<uint32_t>::iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      uint32_t detid = (*detid_iterator);
      // remove any eventual zero elements - there should be none, but just in case
      if(detid == 0) {
	SelectedDetIds.erase(detid_iterator);
        continue;
      }

      ModMEs mod_single;

      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(detid); // pass the detid to this method
      if (reset_each_run) ResetModuleMEs(detid);
      createModuleMEs(mod_single, detid);
      // append to ModuleMEMap
      ModuleMEMap.insert( std::make_pair(detid, mod_single));

      // Created Layer Level MEs if thet=y are npt created already
      std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid);
      if (DetectedLayers.find(det_layer_pair) == DetectedLayers.end()){
	DetectedLayers[det_layer_pair]=true;
	// book Layer plots      
	folder_organizer.setLayerFolder(detid,det_layer_pair.second); 
	createLayerMEs(detid);
      }    
    
    }//end of loop over detectors

  }//end of if

}//end of method



//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;

  runNb   = iEvent.id().run();
  //   eventNb = iEvent.id().event();
  eventNb++;
  //std::cout << " run " << iEvent.id().run() << runNb << " event " << iEvent.id().event() << eventNb << std::endl;

  edm::ESHandle<SiStripNoises> noiseHandle;
  iSetup.get<SiStripNoisesRcd>().get(noiseHandle);

  edm::ESHandle<SiStripGain> gainHandle;
  iSetup.get<SiStripGainRcd>().get(gainHandle);

  std::string quality_label  = conf_.getParameter<std::string>("StripQualityLabel");
  edm::ESHandle<SiStripQuality> qualityHandle;
  iSetup.get<SiStripQualityRcd>().get(quality_label,qualityHandle);

  iSetup.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);

  // retrieve producer name of input StripClusterCollection
  std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
  // get collection of DetSetVector of clusters from Event
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByLabel(clusterProducer, cluster_detsetvektor);

  //if (!cluster_detsetvektor.isValid()) std::cout<<" collection not valid"<<std::endl;
  if (!cluster_detsetvektor.isValid()) return;

  // get list of active detectors from SiStripDetCabling 
  std::vector<uint32_t> SelectedDetIds;
  SiStripDetCabling_->addActiveDetectorsRawIds(SelectedDetIds);
  
  // loop over MEs. Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
  for(std::vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
    if (*detid_iterator == 0) continue;
    uint32_t detid = (*detid_iterator);

    // Get s Module level MEs
    std::map<uint32_t, ModMEs>::const_iterator imodME = ModuleMEMap.find(detid);
    ModMEs mod_single;
    if (imodME != ModuleMEMap.end()) mod_single = imodME->second;

    // Get Layer level MEs
    std::string label;
    getLayerLabel(detid, label);
    SiStripHistoId hidmanager1;
    std::string layer_id= hidmanager1.createHistoLayer("","layer",label,"");
    LayerMEs layer_single;
    std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEMap.find(layer_id);
    if(iLayerME!=LayerMEMap.end()) layer_single = iLayerME->second; 
    
    // get from DetSetVector the DetSet of clusters belonging to one detid - first make sure there exists clusters with this id
    
    edmNew::DetSetVector<SiStripCluster>::const_iterator isearch = cluster_detsetvektor->find(detid); // search  clusters of detid
    
    
    if(isearch==cluster_detsetvektor->end()){
      if(moduleswitchncluson && (mod_single.NumberOfClusters)){
	(mod_single.NumberOfClusters)->Fill(0.); // no clusters for this detector module,fill histogram with 0
      }
      continue; // no clusters for this detid => jump to next step of loop
    }
    
    //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
    edmNew::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
    
    if(moduleswitchncluson && (mod_single.NumberOfClusters != NULL)){ // nr. of clusters per module
      (mod_single.NumberOfClusters)->Fill(static_cast<float>(cluster_detset.size()));
    }
    
    
    short total_clusterized_strips = 0;
    //
    
    SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detid);
    SiStripApvGain::Range detGainRange = gainHandle->getRange(detid); 
    SiStripQuality::Range qualityRange = qualityHandle->getRange(detid);
    
    for(edmNew::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.begin(); clusterIter!= cluster_detset.end(); clusterIter++){
      const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
      const std::vector<uint8_t>& ampls = clusterIter->amplitudes();
      // cluster position
      float cluster_position = clusterIter->barycenter();
      // width defined as nr. of strips that belong to cluster
      short cluster_width    = ampls.size(); 
      // add nr of strips of this cluster to total nr. of clusterized strips
      total_clusterized_strips = total_clusterized_strips + cluster_width; 
      
      // cluster signal and noise from the amplitudes
      float cluster_signal = 0.0;
      float cluster_noise  = 0.0;
      int nrnonzeroamplitudes = 0;
      float noise2 = 0.0;
      float noise  = 0.0;
      for(uint iamp=0; iamp<ampls.size(); iamp++){
	if(ampls[iamp]>0){ // nonzero amplitude
	  cluster_signal += ampls[iamp];
	  try{
	    if(!qualityHandle->IsStripBad(qualityRange, clusterIter->firstStrip()+iamp)){
	      noise = noiseHandle->getNoise(clusterIter->firstStrip()+iamp,detNoiseRange)/gainHandle->getStripGain(clusterIter->firstStrip()+iamp, detGainRange);
	    }
	  }
	  catch(cms::Exception& e){
	    edm::LogError("SiStripTkDQM|SiStripMonitorCluster|DB")<<" cms::Exception:  detid="<<detid<<" firstStrip="<<clusterIter->firstStrip()<<" iamp="<<iamp<<e.what();
	  }
	  noise2 += noise*noise;
	  nrnonzeroamplitudes++;
	}
      } // End loop over cluster amplitude
      if (nrnonzeroamplitudes > 0) cluster_noise = sqrt(noise2/nrnonzeroamplitudes);
      
      if( ps.getParameter<bool>("On") &&
	  (cluster_signal/cluster_noise < ps.getParameter<double>("minStoN") ||
	   cluster_signal/cluster_noise > ps.getParameter<double>("maxStoN") ||
	   cluster_width < ps.getParameter<double>("minWidth") ||
	   cluster_width  > ps.getParameter<double>("maxWidth")) ) continue;  
      
      ClusterProperties cluster_properties;
      cluster_properties.charge    = cluster_signal;
      cluster_properties.position  = cluster_position;
      cluster_properties.width     = cluster_width;
      cluster_properties.noise     = cluster_noise;
      
      // Fill Module Level MEs
      if (imodME != ModuleMEMap.end()) fillModuleMEs(mod_single, cluster_properties);

      // Fill Layer Level MEs
      if (iLayerME!=LayerMEMap.end()) fillLayerMEs(layer_single, cluster_properties);
    }
    
    if(mod_single.NrOfClusterizedStrips && imodME != ModuleMEMap.end()){ // nr of clusterized strips
      mod_single.NrOfClusterizedStrips->Fill(static_cast<float>(total_clusterized_strips));
    }
    
    short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs
    float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
    if(moduleswitchlocaloccupancy && imodME != ModuleMEMap.end()){ // Occupancy
      mod_single.ModuleLocalOccupancy->Fill(local_occupancy);
    }
    
    if (layerswitchlocaloccupancy && (iLayerME!=LayerMEMap.end())) {
      fillME(layer_single.LayerLocalOccupancy,local_occupancy);
      if (createTrendMEs) fillTrend(layer_single.LayerLocalOccupancyTrend,local_occupancy);
    }
    
  }//end of loop over MEs
  
}
//
// -- EndJob
//
void SiStripMonitorCluster::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){

    std::ofstream monitor_summary("monitor_cluster_summary.txt");
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    monitor_summary<<"SiStripMonitorCluster::endJob ModuleMEMap.size()="<<ModuleMEMap.size()<<std::endl;

    for(std::map<uint32_t, ModMEs>::const_iterator idet = ModuleMEMap.begin(); idet!= ModuleMEMap.end(); idet++ ){

      monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"      ++++++detid  "<<idet->first<<std::endl<<std::endl;

      if(moduleswitchncluson) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ NumberOfClusters "<<(idet->second).NumberOfClusters->getEntries()<<" "<<(idet->second).NumberOfClusters->getMean()<<" "<<(idet->second).NumberOfClusters->getRMS()<<std::endl;
      }

      if(moduleswitchclusposon) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterPosition "<<(idet->second).ClusterPosition->getEntries()<<" "<<(idet->second).ClusterPosition->getMean()<<" "<<(idet->second).ClusterPosition->getRMS()<<std::endl;
      }

      if(moduleswitchcluswidthon) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterWidth "<<(idet->second).ClusterWidth->getEntries()<<" "<<(idet->second).ClusterWidth->getMean()<<" "<<(idet->second).ClusterWidth->getRMS()<<std::endl;
      }

      if(moduleswitchcluschargeon) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterCharge "<<(idet->second).ClusterCharge->getEntries()<<" "<<(idet->second).ClusterCharge->getMean()<<" "<<(idet->second).ClusterCharge->getRMS()<<std::endl;
      }

      if(moduleswitchclusnoiseon) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterNoise "<<(idet->second).ClusterNoise->getEntries()<<" "<<(idet->second).ClusterNoise->getMean()<<" "<<(idet->second).ClusterNoise->getRMS()<<std::endl;
      }

      if(moduleswitchclusstonon) {
	monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterSignalOverNoise "<<(idet->second).ClusterSignalOverNoise->getEntries()<<" "<<(idet->second).ClusterSignalOverNoise->getMean()<<" "<<(idet->second).ClusterSignalOverNoise->getRMS()<<std::endl;
      }

      monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ModuleLocalOccupancy "<<(idet->second).ModuleLocalOccupancy->getEntries()<<" "<<(idet->second).ModuleLocalOccupancy->getMean()<<" "<<(idet->second).ModuleLocalOccupancy->getRMS()<<std::endl;

      monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ NrOfClusterizedStrips "<<(idet->second).NrOfClusterizedStrips->getEntries()<<" "<<(idet->second).NrOfClusterizedStrips->getMean()<<" "<<(idet->second).NrOfClusterizedStrips->getRMS()<<std::endl;
 
    }

    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  
    // save histos in a file
    dqmStore_->save(outputFileName);
    
  }//end of if

}



// -- Reset MEs
//------------------------------------------------------------------------------
void SiStripMonitorCluster::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = ModuleMEMap.find(idet);
  ModMEs mod_me = pos->second;

  if (moduleswitchncluson)            mod_me.NumberOfClusters->Reset();
  if (moduleswitchclusposon)          mod_me.ClusterPosition->Reset();
  if (moduleswitchcluswidthon)        mod_me.ClusterWidth->Reset();
  if (moduleswitchcluschargeon)       mod_me.ClusterCharge->Reset();
  if (moduleswitchclusnoiseon)        mod_me.ClusterNoise->Reset();
  if (moduleswitchclusstonon)         mod_me.ClusterSignalOverNoise->Reset();
  if (moduleswitchlocaloccupancy)     mod_me.ModuleLocalOccupancy->Reset();
  if (moduleswitchnrclusterizedstrip) mod_me.NrOfClusterizedStrips->Reset(); 
}
//
// -- Create Module Level MEs
//
void SiStripMonitorCluster::createModuleMEs(ModMEs& mod_single, uint32_t detid) {

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  std::string hid;
  
  //nr. of clusters per module
  if(moduleswitchncluson) {
    hid = hidmanager.createHistoId("NumberOfClusters","det",detid);
    mod_single.NumberOfClusters = bookME1D("TH1nClusters", hid.c_str());
    dqmStore_->tag(mod_single.NumberOfClusters, detid);
    mod_single.NumberOfClusters->setAxisTitle("number of clusters in one detector module");
    mod_single.NumberOfClusters->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
  
  //ClusterPosition
  if(moduleswitchclusposon) {
    short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs    
    hid = hidmanager.createHistoId("ClusterPosition","det",detid);
    mod_single.ClusterPosition = dqmStore_->book1D(hid, hid, total_nr_strips, 0.5, total_nr_strips+0.5);
    dqmStore_->tag(mod_single.ClusterPosition, detid);
    mod_single.ClusterPosition->setAxisTitle("cluster position [strip number +0.5]");
  }
  
  //ClusterWidth
  if(moduleswitchcluswidthon) {
    hid = hidmanager.createHistoId("ClusterWidth","det",detid);
    mod_single.ClusterWidth = bookME1D("TH1ClusterWidth", hid.c_str());
    dqmStore_->tag(mod_single.ClusterWidth, detid);
    mod_single.ClusterWidth->setAxisTitle("cluster width [nr strips]");
  }
  
  //ClusterCharge
  if(moduleswitchcluschargeon) {
    hid = hidmanager.createHistoId("ClusterCharge","det",detid);
    mod_single.ClusterCharge = bookME1D("TH1ClusterCharge", hid.c_str());
    dqmStore_->tag(mod_single.ClusterCharge, detid);
    mod_single.ClusterCharge->setAxisTitle("cluster charge [ADC]");
  }
  
  //ClusterNoise
  if(moduleswitchclusnoiseon) {
    hid = hidmanager.createHistoId("ClusterNoise","det",detid);
    mod_single.ClusterNoise = bookME1D("TH1ClusterNoise", hid.c_str());
    dqmStore_->tag(mod_single.ClusterNoise, detid);
    mod_single.ClusterNoise->setAxisTitle("cluster noise");
  }
  
  //ClusterSignalOverNoise
  if(moduleswitchclusstonon) {
    hid = hidmanager.createHistoId("ClusterSignalOverNoise","det",detid);
    mod_single.ClusterSignalOverNoise = bookME1D("TH1ClusterStoN", hid.c_str());
    dqmStore_->tag(mod_single.ClusterSignalOverNoise, detid);
    mod_single.ClusterSignalOverNoise->setAxisTitle("ratio of signal to noise for each cluster");
  }
  
  //ModuleLocalOccupancy
  if (moduleswitchlocaloccupancy) {
    hid = hidmanager.createHistoId("ClusterLocalOccupancy","det",detid);
    mod_single.ModuleLocalOccupancy = bookME1D("TH1ModuleLocalOccupancy", hid.c_str());
    dqmStore_->tag(mod_single.ModuleLocalOccupancy, detid);
    mod_single.ModuleLocalOccupancy->setAxisTitle("module local occupancy [% of clusterized strips]");
  }
  
  //NrOfClusterizedStrips
  if (moduleswitchnrclusterizedstrip) {
    hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",detid);
    mod_single.NrOfClusterizedStrips = bookME1D("TH1NrOfClusterizedStrips", hid.c_str());
    dqmStore_->tag(mod_single.NrOfClusterizedStrips, detid);
    mod_single.NrOfClusterizedStrips->setAxisTitle("number of clusterized strips");
  }
}  
//
// -- Create Module Level MEs
//  
void SiStripMonitorCluster::createLayerMEs(uint32_t detid) {
  std::string label;
  getLayerLabel(detid, label);
  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoLayer("","layer",label,"");
  std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEMap.find(hid);
  if(iLayerME==LayerMEMap.end()){
    LayerMEs layerMEs; 

    //Cluster Width
    if(layerswitchcluswidthon) {
      layerMEs.LayerClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoLayer("Summary_ClusterWidth","layer",label,"").c_str()); 
      if (createTrendMEs) layerMEs.LayerClusterWidthTrend=bookMETrend("TH1ClusterWidth", hidmanager.createHistoLayer("Trend_ClusterWidth","layer",label,"").c_str()); 
    }

    //Cluster Noise
    if(layerswitchclusnoiseon) {
      layerMEs.LayerClusterNoise=bookME1D("TH1ClusterNoise", hidmanager.createHistoLayer("Summary_ClusterNoise","layer",label,"").c_str()); 
      if (createTrendMEs) layerMEs.LayerClusterNoiseTrend=bookMETrend("TH1ClusterNoise", hidmanager.createHistoLayer("Trend_ClusterNoise","layer",label,"").c_str()); 
    }

    //Cluster Charge
    if(layerswitchcluschargeon) {
      layerMEs.LayerClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoLayer("Summary_ClusterCharge","layer",label,"").c_str());
       if (createTrendMEs) layerMEs.LayerClusterChargeTrend=bookMETrend("TH1ClusterCharge", hidmanager.createHistoLayer("Trend_ClusterCharge","layer",label,"").c_str());
    }

    //Cluster StoN
    if(layerswitchclusstonon) {
      layerMEs.LayerClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoLayer("Summary_ClusterSignalOverNoise","layer",label,"").c_str());
      if (createTrendMEs) layerMEs.LayerClusterStoNTrend=bookMETrend("TH1ClusterStoN", hidmanager.createHistoLayer("Trend_ClusterSignalOverNoise","layer",label,"").c_str());
    }

    //Cluster Occupancy
    if(layerswitchlocaloccupancy) {
      layerMEs.LayerLocalOccupancy=bookME1D("TH1ModuleLocalOccupancy", hidmanager.createHistoLayer("Summary_ClusterLocalOccupancy","layer",label,"").c_str());  
      if (createTrendMEs) layerMEs.LayerLocalOccupancyTrend=bookME1D("TH1ModuleLocalOccupancy", hidmanager.createHistoLayer("Trend_ClusterLocalOccupancy","layer",label,"").c_str());  
      
    }

    LayerMEMap[hid]=layerMEs;
  }

}
//
// -- Fill Module Level Histograms
//
void SiStripMonitorCluster::fillModuleMEs(ModMEs& mod_mes, ClusterProperties& cluster) {
  
  if(moduleswitchclusposon && (mod_mes.ClusterPosition)) // position of cluster
    (mod_mes.ClusterPosition)->Fill(cluster.position);

  if(moduleswitchcluswidthon && (mod_mes.ClusterWidth)) // width of cluster
    (mod_mes.ClusterWidth)->Fill(static_cast<float>(cluster.width));
 
  if(moduleswitchclusstonon && (mod_mes.ClusterSignalOverNoise)) {// SignalToNoise
    if (cluster.noise > 0) 
      (mod_mes.ClusterSignalOverNoise)->Fill(cluster.charge/cluster.noise);
  }

  if(moduleswitchclusnoiseon && (mod_mes.ClusterNoise))  // Noise
    (mod_mes.ClusterNoise)->Fill(cluster.noise);

  if(moduleswitchcluschargeon && (mod_mes.ClusterCharge)) // charge of cluster
    (mod_mes.ClusterCharge)->Fill(cluster.charge);
  
} 
//
// -- Fill Layer Level MEs
//
void SiStripMonitorCluster::fillLayerMEs(LayerMEs& layerMEs, ClusterProperties& cluster) { 
  if(layerswitchclusstonon) {
    fillME(layerMEs.LayerClusterStoN  ,cluster.charge/cluster.noise);
    if (createTrendMEs) fillTrend(layerMEs.LayerClusterStoNTrend,cluster.charge/cluster.noise);
  }
  
  if(layerswitchcluschargeon) {
    fillME(layerMEs.LayerClusterCharge,cluster.charge);
    if (createTrendMEs) fillTrend(layerMEs.LayerClusterChargeTrend,cluster.charge);
  }
  
  if(layerswitchclusnoiseon) {
    fillME(layerMEs.LayerClusterNoise ,cluster.noise);
    if (createTrendMEs) fillTrend(layerMEs.LayerClusterNoiseTrend,cluster.noise);
  }
  
  if(layerswitchcluswidthon) {
    fillME(layerMEs.LayerClusterWidth ,cluster.width);
    if (createTrendMEs) fillTrend(layerMEs.LayerClusterWidthTrend,cluster.width);
  }
}
//
// -- Fill Trend
//
void SiStripMonitorCluster::fillTrend(MonitorElement* me ,float value)
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
MonitorElement* SiStripMonitorCluster::bookMETrend(const char* ParameterSetLabel, const char* HistoName)
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
MonitorElement* SiStripMonitorCluster::bookME1D(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dqmStore_->book1D(HistoName,HistoName,
			   Parameters.getParameter<int32_t>("Nbinx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax")
			   );
}
//-------------------------------------------------------------------------------------------
void SiStripMonitorCluster::getLayerLabel(uint32_t detid, std::string& label) {
  StripSubdetector subdet(detid);
  std::ostringstream label_str;

  if(subdet.subdetId() == StripSubdetector::TIB ){
    // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(detid);
    label_str << "TIB__layer__" << tib1.layer();
  }else if(subdet.subdetId() == StripSubdetector::TID){
    // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(detid);
    label_str << "TID__side__" << tid1.side() << "__wheel__" << tid1.wheel();
  }else if(subdet.subdetId() == StripSubdetector::TOB){
    // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(detid);
    label_str << "TOB__layer__" << tob1.layer();
  }else if(subdet.subdetId() == StripSubdetector::TEC) {
    // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(detid);
    label_str << "TEC__side__"<<tec1.side() << "__wheel__" << tec1.wheel();
  }else{
    // ---------------------------  ???  --------------------------- //
    edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdet.subdetId()<<" no folder set!"<<std::endl;
    label_str << "";
  }
  label = label_str.str();
}

    
