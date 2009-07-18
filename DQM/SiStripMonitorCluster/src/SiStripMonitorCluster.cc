// -*- C++ -*-
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc
 */
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
// $Id: SiStripMonitorCluster.cc,v 1.63 2009/07/01 18:04:42 borrell Exp $
#include <vector>
#include <numeric>
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
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

  edm::ParameterSet ParametersClusterStoNVsPos =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoNVsPos");
  layerswitchclusstonVsposon = ParametersClusterStoNVsPos.getParameter<bool>("layerswitchon");
  moduleswitchclusstonVsposon = ParametersClusterStoNVsPos.getParameter<bool>("moduleswitchon");
  
  edm::ParameterSet ParametersClusterPos =  conf_.getParameter<edm::ParameterSet>("TH1ClusterPos");
  layerswitchclusposon = ParametersClusterPos.getParameter<bool>("layerswitchon");
  moduleswitchclusposon = ParametersClusterPos.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersClusterDigiPos =  conf_.getParameter<edm::ParameterSet>("TH1ClusterDigiPos");
  layerswitchclusdigiposon = ParametersClusterDigiPos.getParameter<bool>("layerswitchon");
  moduleswitchclusdigiposon = ParametersClusterDigiPos.getParameter<bool>("moduleswitchon");
  
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

  edm::ParameterSet ParametersClusterProf = conf_.getParameter<edm::ParameterSet>("TProfNumberOfCluster");
  layerswitchnumclusterprofon = ParametersClusterProf.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterWidthProf = conf_.getParameter<edm::ParameterSet>("TProfClusterWidth");
  layerswitchclusterwidthprofon = ParametersClusterWidthProf.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTotClusterProf = conf_.getParameter<edm::ParameterSet>("TProfTotalNumberOfClusters");
  subdetswitchtotclusterprofon = ParametersTotClusterProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersTotClusterTH1 = conf_.getParameter<edm::ParameterSet>("TH1TotalNumberOfClusters");
  subdetswitchtotclusterth1on = ParametersTotClusterTH1.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersClusterApvProf = conf_.getParameter<edm::ParameterSet>("TProfClustersApvCycle");
  subdetswitchclusterapvprofon = ParametersClusterApvProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersClustersApvTH2 = conf_.getParameter<edm::ParameterSet>("TH2ClustersApvCycle");
  subdetswitchapvcycleth2on = ParametersClustersApvTH2.getParameter<bool>("subdetswitchon");

  clustertkhistomapon = conf_.getParameter<bool>("TkHistoMap_On");
  createTrendMEs = conf_.getParameter<bool>("CreateTrendMEs");
  Mod_On_ = conf_.getParameter<bool>("Mod_On");
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
    
    SiStripSubStructure substructure;

    SiStripFolderOrganizer folder_organizer;
    folder_organizer.setSiStripFolder();

    // Create TkHistoMap for Digi
    if (clustertkhistomapon) tkmapcluster = new TkHistoMap("SiStrip/TkHistoMap","TkHMap_NumberOfCluster",0.,1);

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster")<<"nr. of activeDets:  "<<activeDets.size();
    for(std::vector<uint32_t>::iterator detid_iterator = activeDets.begin(); detid_iterator!=activeDets.end(); detid_iterator++){
      uint32_t detid = (*detid_iterator);
      // remove any eventual zero elements - there should be none, but just in case
      if(detid == 0) {
	activeDets.erase(detid_iterator);
        continue;
      }

      if (Mod_On_) {
	ModMEs mod_single;
	// set appropriate folder using SiStripFolderOrganizer
	folder_organizer.setDetectorFolder(detid); // pass the detid to this method
	if (reset_each_run) ResetModuleMEs(detid);
	createModuleMEs(mod_single, detid);
	// append to ModuleMEMap
	ModuleMEMap.insert( std::make_pair(detid, mod_single));
      }

      // Created Layer Level MEs if they are not created already
      std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid);
      if (DetectedLayers.find(det_layer_pair) == DetectedLayers.end()){
        DetectedLayers[det_layer_pair]=true;

        int32_t lnumber = det_layer_pair.second;
	std::vector<uint32_t> layerDetIds;
        if (det_layer_pair.first == "TIB") {
          substructure.getTIBDetectors(activeDets,layerDetIds,lnumber,0,0,0);
          if (SubDetMEsMap.find("TIB") == SubDetMEsMap.end()) createSubDetMEs("TIB");
        } else if (det_layer_pair.first == "TOB") {
          substructure.getTOBDetectors(activeDets,layerDetIds,lnumber,0,0);
          if (SubDetMEsMap.find("TOB") == SubDetMEsMap.end()) createSubDetMEs("TOB");
        } else if (det_layer_pair.first == "TID" && lnumber > 0) {
          substructure.getTIDDetectors(activeDets,layerDetIds,2,abs(lnumber),0,0);
          if (SubDetMEsMap.find("TID") == SubDetMEsMap.end()) createSubDetMEs("TID");
        } else if (det_layer_pair.first == "TID" && lnumber < 0) {
          substructure.getTIDDetectors(activeDets,layerDetIds,1,abs(lnumber),0,0);
          if (SubDetMEsMap.find("TID") == SubDetMEsMap.end()) createSubDetMEs("TID");
        } else if (det_layer_pair.first == "TEC" && lnumber > 0) {
          substructure.getTECDetectors(activeDets,layerDetIds,2,abs(lnumber),0,0,0,0);
          if (SubDetMEsMap.find("TEC") == SubDetMEsMap.end()) createSubDetMEs("TEC");
        } else if (det_layer_pair.first == "TEC" && lnumber < 0) {
          substructure.getTECDetectors(activeDets,layerDetIds,1,abs(lnumber),0,0,0,0);
          if (SubDetMEsMap.find("TEC") == SubDetMEsMap.end()) createSubDetMEs("TEC");
        }

	SiStripHistoId hidmanager;
	std::string label = hidmanager.getSubdetid(detid,false);
        if (label.size() > 0) {
          LayerDetMap[label] = layerDetIds;
          // book Layer plots
          folder_organizer.setLayerFolder(detid,det_layer_pair.second);
          createLayerMEs(label, layerDetIds.size());
        }
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
  float iOrbitSec = iEvent.orbitNumber()/11223.0;
  int bx = iEvent.bunchCrossing();
  long long tbx = (long long)iEvent.orbitNumber() * 3564 + bx; 

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
 
  int nTotClusterTIB = 0;
  int nTotClusterTOB = 0;
  int nTotClusterTEC = 0;
  int nTotClusterTID = 0;
  //  int nTotCluster    = 0;

  bool found_layer_me = false;
  for (std::map<std::string, std::vector< uint32_t > >::const_iterator iterLayer = LayerDetMap.begin();
       iterLayer != LayerDetMap.end(); iterLayer++) {
    
    std::string layer_label = iterLayer->first;
    
    std::vector< uint32_t > layer_dets = iterLayer->second;
    int ncluster_layer = 0;
    std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEMap.find(layer_label);
    
    //get Layer MEs 
    LayerMEs layer_single;
    if(iLayerME != LayerMEMap.end()) {
       layer_single = iLayerME->second; 
       found_layer_me = true;
     } 

    bool found_module_me = false;
    uint16_t iDet = 0;
    // loop over all modules in the layer
    for (std::vector< uint32_t >::const_iterator iterDets = layer_dets.begin() ; 
	 iterDets != layer_dets.end() ; iterDets++) {
      iDet++;
      // detid and type of ME
      uint32_t detid = (*iterDets);
      
      // DetId and corresponding set of MEs
      ModMEs mod_single;
      if (Mod_On_) {
	std::map<uint32_t, ModMEs >::iterator imodME = ModuleMEMap.find(detid);
	if (imodME != ModuleMEMap.end()) {
	  mod_single = imodME->second;
	  found_module_me = true;
	} 
      } else found_module_me = false;

      edmNew::DetSetVector<SiStripCluster>::const_iterator isearch = cluster_detsetvektor->find(detid); // search  clusters of detid
    
      if(isearch==cluster_detsetvektor->end()){
	if(found_module_me && moduleswitchncluson && (mod_single.NumberOfClusters)){
	  (mod_single.NumberOfClusters)->Fill(0.); // no clusters for this detector module,fill histogram with 0
	}
	if(clustertkhistomapon) tkmapcluster->fill(detid,0.);
	if (found_layer_me && layerswitchnumclusterprofon) layer_single.LayerNumberOfClusterProfile->Fill(iDet, 0.0);
	continue; // no clusters for this detid => jump to next step of loop
      }
      
      //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
      edmNew::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
      
      // Filling TkHistoMap with number of clusters for each module 
      if(clustertkhistomapon) {
	tkmapcluster->fill(detid,static_cast<float>(cluster_detset.size()));
      }

      if(moduleswitchncluson && found_module_me && (mod_single.NumberOfClusters != NULL)){ // nr. of clusters per module
	(mod_single.NumberOfClusters)->Fill(static_cast<float>(cluster_detset.size()));
      }
      if (found_layer_me && layerswitchnumclusterprofon) 
	layer_single.LayerNumberOfClusterProfile->Fill(iDet, static_cast<float>(cluster_detset.size()));
      ncluster_layer +=  cluster_detset.size();
      
      short total_clusterized_strips = 0;
      
      SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detid);
      SiStripApvGain::Range detGainRange = gainHandle->getRange(detid); 
      SiStripQuality::Range qualityRange = qualityHandle->getRange(detid);
      
      for(edmNew::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.begin(); clusterIter!= cluster_detset.end(); clusterIter++){
	const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
	const std::vector<uint8_t>& ampls = clusterIter->amplitudes();
	// cluster position
	float cluster_position = clusterIter->barycenter();
	// start defined as nr. of first strip beloning to the cluster
	short cluster_start    = clusterIter->firstStrip();
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
	cluster_properties.start     = cluster_start;
	cluster_properties.width     = cluster_width;
	cluster_properties.noise     = cluster_noise;
	
	// Fill Module Level MEs
	if (found_module_me) fillModuleMEs(mod_single, cluster_properties);
	
	// Fill Layer Level MEs
	if (found_layer_me) {
          fillLayerMEs(layer_single, cluster_properties, iOrbitSec);
	  if (layerswitchclusterwidthprofon) 
	    layer_single.LayerClusterWidthProfile->Fill(iDet, cluster_width);
	}
      } // end loop over clusters
      
      short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs
      float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
      if (found_module_me) {
	if(moduleswitchnrclusterizedstrip && mod_single.NrOfClusterizedStrips ){ // nr of clusterized strips
	  mod_single.NrOfClusterizedStrips->Fill(static_cast<float>(total_clusterized_strips));
	}
	
	if(moduleswitchlocaloccupancy && mod_single.ModuleLocalOccupancy ){ // Occupancy
	  mod_single.ModuleLocalOccupancy->Fill(local_occupancy);
	}
      }
      if (layerswitchlocaloccupancy && found_layer_me && layer_single.LayerLocalOccupancy) {
	fillME(layer_single.LayerLocalOccupancy,local_occupancy);
	if (createTrendMEs) fillME(layer_single.LayerLocalOccupancyTrend,iOrbitSec,local_occupancy);
      }
    }
    if (layer_label.find("TIB") != std::string::npos)      nTotClusterTIB += ncluster_layer;
    else if (layer_label.find("TOB") != std::string::npos) nTotClusterTOB += ncluster_layer;
    else if (layer_label.find("TEC") != std::string::npos) nTotClusterTEC += ncluster_layer;        
    else if (layer_label.find("TID") != std::string::npos) nTotClusterTID += ncluster_layer;        
  }
  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin();
       it != SubDetMEsMap.end(); it++) {
    SubDetMEs subdetmes; 
    subdetmes  = it->second;
    if (subdetswitchtotclusterprofon) {
      if (it->first == "TIB") subdetmes.SubDetTotClusterProf->Fill(iOrbitSec,nTotClusterTIB);
      else if (it->first == "TOB") subdetmes.SubDetTotClusterProf->Fill(iOrbitSec,nTotClusterTOB);
      else if (it->first == "TID") subdetmes.SubDetTotClusterProf->Fill(iOrbitSec,nTotClusterTID);
      else if (it->first == "TEC") subdetmes.SubDetTotClusterProf->Fill(iOrbitSec,nTotClusterTEC);      
    }
    if (subdetswitchclusterapvprofon) {
      if (it->first == "TIB") subdetmes.SubDetClusterApvProf->Fill(tbx%70,nTotClusterTIB);
      else if (it->first == "TOB") subdetmes.SubDetClusterApvProf->Fill(tbx%70,nTotClusterTOB);
      else if (it->first == "TID") subdetmes.SubDetClusterApvProf->Fill(tbx%70,nTotClusterTID);
      else if (it->first == "TEC") subdetmes.SubDetClusterApvProf->Fill(tbx%70,nTotClusterTEC);      
    }
    if (subdetswitchapvcycleth2on) {
      if (it->first == "TIB") subdetmes.SubDetClusterApvTH2->Fill(tbx%70,nTotClusterTIB);
      else if (it->first == "TOB") subdetmes.SubDetClusterApvTH2->Fill(tbx%70,nTotClusterTOB);
      else if (it->first == "TID") subdetmes.SubDetClusterApvTH2->Fill(tbx%70,nTotClusterTID);
      else if (it->first == "TEC") subdetmes.SubDetClusterApvTH2->Fill(tbx%70,nTotClusterTEC);      
    }
    if (subdetswitchtotclusterth1on) {
      if (it->first == "TIB") subdetmes.SubDetTotClusterTH1->Fill(nTotClusterTIB);
      else if (it->first == "TOB") subdetmes.SubDetTotClusterTH1->Fill(nTotClusterTOB);
      else if (it->first == "TID") subdetmes.SubDetTotClusterTH1->Fill(nTotClusterTID);
      else if (it->first == "TEC") subdetmes.SubDetTotClusterTH1->Fill(nTotClusterTEC);      
    }
  }
}
//
// -- EndJob
//
void SiStripMonitorCluster::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
 
  // save histos in a file
  if(outputMEsInRootFile) dqmStore_->save(outputFileName);
}
//
// -- Reset MEs
//------------------------------------------------------------------------------
void SiStripMonitorCluster::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = ModuleMEMap.find(idet);
  ModMEs mod_me = pos->second;

  if (moduleswitchncluson)            mod_me.NumberOfClusters->Reset();
  if (moduleswitchclusposon)          mod_me.ClusterPosition->Reset();
  if (moduleswitchclusdigiposon)            mod_me.ClusterDigiPosition->Reset();
  if (moduleswitchclusstonVsposon)    mod_me.ClusterSignalOverNoiseVsPos->Reset();
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

  //ClusterDigiPosition
  if(moduleswitchclusdigiposon) {
    short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs    
    hid = hidmanager.createHistoId("ClusterDigiPosition","det",detid);
    mod_single.ClusterDigiPosition = dqmStore_->book1D(hid, hid, total_nr_strips, 0.5, total_nr_strips+0.5);
    dqmStore_->tag(mod_single.ClusterDigiPosition, detid);
    mod_single.ClusterDigiPosition->setAxisTitle("digi in cluster position [strip number +0.5]");
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

  //ClusterSignalOverNoiseVsPos
  if(moduleswitchclusstonVsposon) {
    hid = hidmanager.createHistoId("ClusterSignalOverNoiseVsPos","det",detid);
    Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoNVsPos");
    mod_single.ClusterSignalOverNoiseVsPos= dqmStore_->bookProfile(hid.c_str(),hid.c_str(),
								   Parameters.getParameter<int32_t>("Nbinx"),
								   Parameters.getParameter<double>("xmin"),
								   Parameters.getParameter<double>("xmax"),
								   Parameters.getParameter<int32_t>("Nbiny"),
								   Parameters.getParameter<double>("ymin"),
								   Parameters.getParameter<double>("ymax")
								   );
    dqmStore_->tag(mod_single.ClusterSignalOverNoiseVsPos, detid);
    mod_single.ClusterSignalOverNoiseVsPos->setAxisTitle("pos");
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
void SiStripMonitorCluster::createLayerMEs(std::string label, int ndets) {

  std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEMap.find(label);
  if(iLayerME==LayerMEMap.end()){
    SiStripHistoId hidmanager;
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
      if (createTrendMEs) layerMEs.LayerLocalOccupancyTrend=bookMETrend("TH1ModuleLocalOccupancy", hidmanager.createHistoLayer("Trend_ClusterLocalOccupancy","layer",label,"").c_str());  
      
    }

    // # of Cluster Profile 
    if(layerswitchnumclusterprofon) {
      std::string hid = hidmanager.createHistoLayer("NumberOfClusterProfile","layer",label,"");
      layerMEs.LayerNumberOfClusterProfile = dqmStore_->bookProfile(hid, hid, ndets, 0.5, ndets+0.5,21, -0.5, 20.5);      
    }

    // Cluster Width Profile 
    if(layerswitchclusterwidthprofon) {
      std::string hid = hidmanager.createHistoLayer("ClusterWidthProfile","layer",label,"");      
      layerMEs.LayerClusterWidthProfile = dqmStore_->bookProfile(hid, hid, ndets, 0.5, ndets+0.5, 20, -0.5, 19.5);      
    }

    LayerMEMap[label]=layerMEs;
  }

}
//
// -- Create SubDetector MEs
//
void SiStripMonitorCluster::createSubDetMEs(std::string label) {

  std::map<std::string, SubDetMEs>::iterator iSubDetME  = SubDetMEsMap.find(label);
  if(iSubDetME==SubDetMEsMap.end()){
    SubDetMEs subdetMEs; 
    std::string HistoName;

  if (subdetswitchtotclusterprofon){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TProfTotalNumberOfClusters");
    dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "TotalNumberOfClusterProfile__" + label;
    subdetMEs.SubDetTotClusterProf = dqmStore_->bookProfile(HistoName,HistoName,
					      Parameters.getParameter<int32_t>("Nbins"),
					      Parameters.getParameter<double>("xmin"),
					      Parameters.getParameter<double>("xmax"),
					      100, //that parameter should not be there !?
					      Parameters.getParameter<double>("ymin"),
					      Parameters.getParameter<double>("ymax"),
					      "" );
  subdetMEs.SubDetTotClusterProf->setAxisTitle("Event Time (Seconds)",1);
  if (subdetMEs.SubDetTotClusterProf->kind() == MonitorElement::DQM_KIND_TPROFILE) subdetMEs.SubDetTotClusterProf->getTH1()->SetBit(TH1::kCanRebin);
  }
    // Number of Cluster vs APV cycle - Profile
    if(subdetswitchclusterapvprofon){
      edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TProfClustersApvCycle");
      dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
      HistoName = "Cluster_vs_ApvCycle_" + label;
      subdetMEs.SubDetClusterApvProf=dqmStore_->bookProfile(HistoName,HistoName,
					      Parameters.getParameter<int32_t>("Nbins"),
					      Parameters.getParameter<double>("xmin"),
					      Parameters.getParameter<double>("xmax"),
					      200, //that parameter should not be there !?
					      Parameters.getParameter<double>("ymin"),
					      Parameters.getParameter<double>("ymax"),
					      "" );
      subdetMEs.SubDetClusterApvProf->setAxisTitle("absolute Bx mod(70)",1);
    }

  if (subdetswitchtotclusterth1on){
    dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "TotalNumberOfCluster__" + label;
    subdetMEs.SubDetTotClusterTH1 = bookME1D("TH1TotalNumberOfClusters",HistoName.c_str());
    subdetMEs.SubDetTotClusterTH1->setAxisTitle("Total number of clusters in subdetector");
    subdetMEs.SubDetTotClusterTH1->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
    // TH2 with number of Clusters vs Bx 
    if(subdetswitchapvcycleth2on){
      edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TH2ClustersApvCycle");
      dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
      HistoName = "Cluster_vs_ApvCycle_2D_" + label;
      // Adjusting the scale for 2D histogram
      double h2ymax = 9999.0;     
      double yfact = Parameters.getParameter<double>("yfactor");
      if(label == "TIB") h2ymax = (6984.*256.)*yfact;
      else if (label == "TID") h2ymax = (2208.*256.)*yfact;
      else if (label == "TOB") h2ymax = (12906.*256.)*yfact;
      else if (label == "TEC") h2ymax = (7552.*2.*256.)*yfact;

      subdetMEs.SubDetClusterApvTH2=dqmStore_->book2D(HistoName,HistoName,
					      Parameters.getParameter<int32_t>("Nbins"),
					      Parameters.getParameter<double>("xmin"),
					      Parameters.getParameter<double>("xmax"),
					      Parameters.getParameter<int32_t>("Nbinsy"),
					      Parameters.getParameter<double>("ymin"),
					      h2ymax);
      subdetMEs.SubDetClusterApvTH2->setAxisTitle("absolute Bx mod(70)",1);
    }
  SubDetMEsMap[label]=subdetMEs;
  }
}

//
// -- Fill Module Level Histograms
//
void SiStripMonitorCluster::fillModuleMEs(ModMEs& mod_mes, ClusterProperties& cluster) {
  
  if(moduleswitchclusposon && (mod_mes.ClusterPosition)) // position of cluster
    (mod_mes.ClusterPosition)->Fill(cluster.position);
  
  // position of digis in cluster
  if(moduleswitchclusdigiposon && (mod_mes.ClusterDigiPosition)) {
    for(int ipos=cluster.start+1; ipos<=cluster.start+cluster.width; ipos++){
      (mod_mes.ClusterDigiPosition)->Fill(ipos);
    }
  }

  if(moduleswitchcluswidthon && (mod_mes.ClusterWidth)) // width of cluster
    (mod_mes.ClusterWidth)->Fill(static_cast<float>(cluster.width));
 
  if(moduleswitchclusstonon && (mod_mes.ClusterSignalOverNoise)) {// SignalToNoise
    if (cluster.noise > 0) 
      (mod_mes.ClusterSignalOverNoise)->Fill(cluster.charge/cluster.noise);
  }

  if(moduleswitchclusstonVsposon && (mod_mes.ClusterSignalOverNoiseVsPos)) {// SignalToNoise
    if (cluster.noise > 0) 
      (mod_mes.ClusterSignalOverNoiseVsPos)->Fill(cluster.position,cluster.charge/cluster.noise);
  }

  if(moduleswitchclusnoiseon && (mod_mes.ClusterNoise))  // Noise
    (mod_mes.ClusterNoise)->Fill(cluster.noise);

  if(moduleswitchcluschargeon && (mod_mes.ClusterCharge)) // charge of cluster
    (mod_mes.ClusterCharge)->Fill(cluster.charge);
  
} 
//
// -- Fill Layer Level MEs
//
void SiStripMonitorCluster::fillLayerMEs(LayerMEs& layerMEs, ClusterProperties& cluster, float timeinorbit) { 
  if(layerswitchclusstonon) {
    fillME(layerMEs.LayerClusterStoN  ,cluster.charge/cluster.noise);
    if (createTrendMEs) fillME(layerMEs.LayerClusterStoNTrend,timeinorbit,cluster.charge/cluster.noise);
  }
  
  if(layerswitchcluschargeon) {
    fillME(layerMEs.LayerClusterCharge,cluster.charge);
    if (createTrendMEs) fillME(layerMEs.LayerClusterChargeTrend,timeinorbit,cluster.charge);
  }
  
  if(layerswitchclusnoiseon) {
    fillME(layerMEs.LayerClusterNoise ,cluster.noise);
    if (createTrendMEs) fillME(layerMEs.LayerClusterNoiseTrend,timeinorbit,cluster.noise);
  }
  
  if(layerswitchcluswidthon) {
    fillME(layerMEs.LayerClusterWidth ,cluster.width);
    if (createTrendMEs) fillME(layerMEs.LayerClusterWidthTrend,timeinorbit,cluster.width);
  }
  
}
//------------------------------------------------------------------------------------------
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
  me->setAxisTitle("Event Time in Seconds",1);
  if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) me->getTH1()->SetBit(TH1::kCanRebin);
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


    
