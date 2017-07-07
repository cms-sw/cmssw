
// -*- C++ -*-
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc
 */
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
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
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TMath.h"
#include <iostream>

//--------------------------------------------------------------------------------------------
SiStripMonitorCluster::SiStripMonitorCluster(const edm::ParameterSet& iConfig)
  : conf_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(false), reset_each_run(false), m_cacheID_(0)
					    //  , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig, consumesCollector()))
{

  // initialize
  passBPTXfilter_ = true;

  // initialize GenericTriggerEventFlag by specific configuration
  // in this way, one can set specific selections for different MEs
  genTriggerEventFlagBPTXfilter_     = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("BPTXfilter")    , consumesCollector(), *this );
  genTriggerEventFlagPixelDCSfilter_ = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("PixelDCSfilter"), consumesCollector(), *this );
  genTriggerEventFlagStripDCSfilter_ = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("StripDCSfilter"), consumesCollector(), *this );

  firstEvent = -1;
  eventNb = 0;

  // Detector Partitions
  SubDetPhasePartMap["TIB"]        = "TI";
  SubDetPhasePartMap["TID__MINUS"] = "TI";
  SubDetPhasePartMap["TID__PLUS"]  = "TI";
  SubDetPhasePartMap["TOB"]        = "TO";
  SubDetPhasePartMap["TEC__MINUS"] = "TM";
  SubDetPhasePartMap["TEC__PLUS"]  = "TP";

  //get on/off option for every cluster from cfi
  edm::ParameterSet ParametersnClusters =  conf_.getParameter<edm::ParameterSet>("TH1nClusters");
  layerswitchncluson = ParametersnClusters.getParameter<bool>("layerswitchon");
  moduleswitchncluson = ParametersnClusters.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersClusterCharge =  conf_.getParameter<edm::ParameterSet>("TH1ClusterCharge");
  layerswitchcluschargeon = ParametersClusterCharge.getParameter<bool>("layerswitchon");
  moduleswitchcluschargeon = ParametersClusterCharge.getParameter<bool>("moduleswitchon");
  subdetswitchcluschargeon = ParametersClusterCharge.getParameter<bool>("subdetswitchon");

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
  subdetswitchcluswidthon = ParametersClusterWidth.getParameter<bool>("subdetswitchon");

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
  subdetswitchtotclusprofon = ParametersTotClusterProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersTotClusterTH1 = conf_.getParameter<edm::ParameterSet>("TH1TotalNumberOfClusters");
  subdetswitchtotclusth1on = ParametersTotClusterTH1.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersClusterApvProf = conf_.getParameter<edm::ParameterSet>("TProfClustersApvCycle");
  subdetswitchapvcycleprofon = ParametersClusterApvProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersClustersApvTH2 = conf_.getParameter<edm::ParameterSet>("TH2ClustersApvCycle");
  subdetswitchapvcycleth2on = ParametersClustersApvTH2.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersApvCycleDBxProf2 = conf_.getParameter<edm::ParameterSet>("TProf2ApvCycleVsDBx");
  subdetswitchapvcycledbxprof2on = ParametersApvCycleDBxProf2.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersDBxCycleProf = conf_.getParameter<edm::ParameterSet>("TProfClustersVsDBxCycle");
  subdetswitchdbxcycleprofon = ParametersDBxCycleProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersCStripVsCPix = conf_.getParameter<edm::ParameterSet>("TH2CStripVsCpixel");
  globalswitchcstripvscpix = ParametersCStripVsCPix.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersMultiplicityRegionsTH1 = conf_.getParameter<edm::ParameterSet>("TH1MultiplicityRegions");
  globalswitchMultiRegions =  ParametersMultiplicityRegionsTH1.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersApvCycleVsDBxGlobalTH2 = conf_.getParameter<edm::ParameterSet>("TH2ApvCycleVsDBxGlobal");
  globalswitchapvcycledbxth2on = ParametersApvCycleVsDBxGlobalTH2.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersNoiseStrip2ApvCycle = conf_.getParameter<edm::ParameterSet>("TH1StripNoise2ApvCycle");
  globalswitchstripnoise2apvcycle = ParametersNoiseStrip2ApvCycle.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersNoiseStrip3ApvCycle = conf_.getParameter<edm::ParameterSet>("TH1StripNoise3ApvCycle");
  globalswitchstripnoise3apvcycle = ParametersNoiseStrip3ApvCycle.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersMainDiagonalPosition = conf_.getParameter<edm::ParameterSet>("TH1MainDiagonalPosition");
  globalswitchmaindiagonalposition= ParametersMainDiagonalPosition.getParameter<bool>("globalswitchon");

  edm::ParameterSet ClusterMultiplicityRegions = conf_.getParameter<edm::ParameterSet>("MultiplicityRegions");
  k0 = ClusterMultiplicityRegions.getParameter<double>("k0");
  q0 = ClusterMultiplicityRegions.getParameter<double>("q0");
  dk0 = ClusterMultiplicityRegions.getParameter<double>("dk0");
  maxClus = ClusterMultiplicityRegions.getParameter<double>("MaxClus");
  minPix = ClusterMultiplicityRegions.getParameter<double>("MinPix");

  edm::ParameterSet ParametersNclusVsCycleTimeProf2D = conf_.getParameter<edm::ParameterSet>("NclusVsCycleTimeProf2D");
  globalswitchnclusvscycletimeprof2don = ParametersNclusVsCycleTimeProf2D.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersClusWidthVsAmpTH2 = conf_.getParameter<edm::ParameterSet>("ClusWidthVsAmpTH2");
  clusterWidth_vs_amplitude_on = ParametersClusWidthVsAmpTH2.getParameter<bool>("globalswitchon");
  layer_clusterWidth_vs_amplitude_on = ParametersClusWidthVsAmpTH2.getParameter<bool>("layerswitchon");
  subdet_clusterWidth_vs_amplitude_on = ParametersClusWidthVsAmpTH2.getParameter<bool>("subdetswitchon");
  module_clusterWidth_vs_amplitude_on = ParametersClusWidthVsAmpTH2.getParameter<bool>("moduleswitchon");

  clustertkhistomapon = conf_.getParameter<bool>("TkHistoMap_On");
  clusterchtkhistomapon = conf_.getParameter<bool>("ClusterChTkHistoMap_On");
  createTrendMEs = conf_.getParameter<bool>("CreateTrendMEs");
  trendVsLs_ = conf_.getParameter<bool>("TrendVsLS");
  Mod_On_ = conf_.getParameter<bool>("Mod_On");
  ClusterHisto_ = conf_.getParameter<bool>("ClusterHisto");

  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");


  // Poducer name of input StripClusterCollection
  clusterProducerStripToken_ = consumes<edmNew::DetSetVector<SiStripCluster> >(conf_.getParameter<edm::InputTag>("ClusterProducerStrip") );
  clusterProducerPixToken_   = consumes<edmNew::DetSetVector<SiPixelCluster> >(conf_.getParameter<edm::InputTag>("ClusterProducerPix") );
  /*
  clusterProducerStrip_ = conf_.getParameter<edm::InputTag>("ClusterProducerStrip");
  clusterProducerPix_ = conf_.getParameter<edm::InputTag>("ClusterProducerPix");
  */
  // SiStrip Quality Label
  qualityLabel_  = conf_.getParameter<std::string>("StripQualityLabel");
  // cluster quality conditions
  edm::ParameterSet cluster_condition = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  applyClusterQuality_ = cluster_condition.getParameter<bool>("On");
  sToNLowerLimit_      = cluster_condition.getParameter<double>("minStoN");
  sToNUpperLimit_      = cluster_condition.getParameter<double>("maxStoN");
  widthLowerLimit_     = cluster_condition.getParameter<double>("minWidth");
  widthUpperLimit_     = cluster_condition.getParameter<double>("maxWidth");

  // Event History Producer
  //  historyProducer_ = conf_.getParameter<edm::InputTag>("HistoryProducer");
  historyProducerToken_ = consumes<EventWithHistory>(conf_.getParameter<edm::InputTag>("HistoryProducer") );
  // Apv Phase Producer
  //  apvPhaseProducer_ = conf_.getParameter<edm::InputTag>("ApvPhaseProducer");
  apvPhaseProducerToken_ = consumes<APVCyclePhaseCollection>(conf_.getParameter<edm::InputTag>("ApvPhaseProducer") );
  // Create DCS Status
  bool checkDCS    = conf_.getParameter<bool>("UseDCSFiltering");
  if (checkDCS) dcsStatus_ = new SiStripDCSStatus(consumesCollector());
  else dcsStatus_ = 0;

}

SiStripMonitorCluster::~SiStripMonitorCluster() {
  if (dcsStatus_)           delete dcsStatus_;
  if (genTriggerEventFlagBPTXfilter_    ) delete genTriggerEventFlagBPTXfilter_;
  if (genTriggerEventFlagPixelDCSfilter_) delete genTriggerEventFlagPixelDCSfilter_;
  if (genTriggerEventFlagStripDCSfilter_) delete genTriggerEventFlagStripDCSfilter_;
}


//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::dqmBeginRun(const edm::Run& run, const edm::EventSetup &es ) {

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlagBPTXfilter_->on() )
    genTriggerEventFlagBPTXfilter_->initRun( run, es );
  if ( genTriggerEventFlagPixelDCSfilter_->on() )
    genTriggerEventFlagPixelDCSfilter_->initRun( run, es );
  if ( genTriggerEventFlagStripDCSfilter_->on() )
    genTriggerEventFlagStripDCSfilter_->initRun( run, es );
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::createMEs(const edm::EventSetup& es , DQMStore::IBooker & ibooker){

  if ( show_mechanical_structure_view ){

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    es.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);

    // get list of active detectors from SiStripDetCabling
    std::vector<uint32_t> activeDets;
    SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);

    SiStripSubStructure substructure;

    SiStripFolderOrganizer folder_organizer;
    folder_organizer.setSiStripFolderName(topFolderName_);
    folder_organizer.setSiStripFolder();


    // Create TkHistoMap for Cluster
    if (clustertkhistomapon) {
      //      std::cout << "[SiStripMonitorCluster::createMEs] topFolderName_: " << topFolderName_ << "     ";
      if ( (topFolderName_ == "SiStrip") or (std::string::npos != topFolderName_.find("HLT")) )
	tkmapcluster = new TkHistoMap(ibooker , topFolderName_,"TkHMap_NumberOfCluster",0.,true);
      else tkmapcluster = new TkHistoMap(ibooker , topFolderName_+"/TkHistoMap","TkHMap_NumberOfCluster",0.,false);
    }
    if (clusterchtkhistomapon) {
     if ( (topFolderName_ == "SiStrip") or (std::string::npos != topFolderName_.find("HLT")) )
       tkmapclusterch = new TkHistoMap(ibooker , topFolderName_,"TkHMap_ClusterCharge",0.,true);
     else tkmapclusterch = new TkHistoMap(ibooker , topFolderName_+"/TkHistoMap","TkHMap_ClusterCharge",0.,false);
    }

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
	folder_organizer.setDetectorFolder(detid, tTopo); // pass the detid to this method
	if (reset_each_run) ResetModuleMEs(detid);
	createModuleMEs(mod_single, detid , ibooker);
	// append to ModuleMEsMap
	ModuleMEsMap.insert( std::make_pair(detid, mod_single));
      }

      // Create Layer Level MEs if they are not created already
      std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid, tTopo);
      SiStripHistoId hidmanager;
      std::string label = hidmanager.getSubdetid(detid,tTopo,false);

      std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
      if(iLayerME==LayerMEsMap.end()) {

        // get detids for the layer
        int32_t lnumber = det_layer_pair.second;
	std::vector<uint32_t> layerDetIds;
        if (det_layer_pair.first == "TIB") {
          substructure.getTIBDetectors(activeDets,layerDetIds,lnumber,0,0,0);
        } else if (det_layer_pair.first == "TOB") {
          substructure.getTOBDetectors(activeDets,layerDetIds,lnumber,0,0);
        } else if (det_layer_pair.first == "TID" && lnumber > 0) {
          substructure.getTIDDetectors(activeDets,layerDetIds,2,abs(lnumber),0,0);
        } else if (det_layer_pair.first == "TID" && lnumber < 0) {
          substructure.getTIDDetectors(activeDets,layerDetIds,1,abs(lnumber),0,0);
        } else if (det_layer_pair.first == "TEC" && lnumber > 0) {
          substructure.getTECDetectors(activeDets,layerDetIds,2,abs(lnumber),0,0,0,0);
        } else if (det_layer_pair.first == "TEC" && lnumber < 0) {
          substructure.getTECDetectors(activeDets,layerDetIds,1,abs(lnumber),0,0,0,0);
        }
	LayerDetMap[label] = layerDetIds;

	// book Layer MEs
	folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second);
	createLayerMEs(label, layerDetIds.size() , ibooker );
      }
      // book sub-detector plots
      auto sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
      if (SubDetMEsMap.find(sdet_pair.second) == SubDetMEsMap.end()){
	ibooker.setCurrentFolder(sdet_pair.first);

	createSubDetMEs(sdet_pair.second , ibooker );
      }
    }//end of loop over detectors

    // Create Global Histogram
    if (globalswitchapvcycledbxth2on) {
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH2Parameters =  conf_.getParameter<edm::ParameterSet>("TH2ApvCycleVsDBxGlobal");
      std::string HistoName = "DeltaBx_vs_ApvCycle";
      GlobalApvCycleDBxTH2 = ibooker.book2D(HistoName,HistoName,
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsx"),
					       GlobalTH2Parameters.getParameter<double>("xmin"),
					       GlobalTH2Parameters.getParameter<double>("xmax"),
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsy"),
					       GlobalTH2Parameters.getParameter<double>("ymin"),
					       GlobalTH2Parameters.getParameter<double>("ymax"));
      GlobalApvCycleDBxTH2->setAxisTitle("APV Cycle (Corrected Absolute Bx % 70)",1);
      GlobalApvCycleDBxTH2->setAxisTitle("Delta Bunch Crossing Cycle",2);

      // plot DeltaBX ***************************
      edm::ParameterSet GlobalTH1Parameters =  conf_.getParameter<edm::ParameterSet>("TH1DBxGlobal");
      HistoName = "DeltaBx";
      GlobalDBxTH1 = ibooker.book1D(HistoName,HistoName,
				    GlobalTH1Parameters.getParameter<int32_t>("Nbinsx"),
				    GlobalTH1Parameters.getParameter<double>("xmin"),
				    GlobalTH1Parameters.getParameter<double>("xmax"));
      GlobalDBxTH1->setAxisTitle("Delta Bunch Crossing",1);



      // plot DeltaBXCycle ***************************
      edm::ParameterSet DBxCycle =  conf_.getParameter<edm::ParameterSet>("TH1DBxCycleGlobal");
      HistoName = "DeltaBxCycle";
      GlobalDBxCycleTH1 = ibooker.book1D(HistoName,HistoName,
				    DBxCycle.getParameter<int32_t>("Nbinsx"),
				    DBxCycle.getParameter<double>("xmin"),
				    DBxCycle.getParameter<double>("xmax"));
      GlobalDBxCycleTH1->setAxisTitle("Delta Bunch Crossing Cycle",1);


    }

    if (globalswitchcstripvscpix) {
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH2Parameters =  conf_.getParameter<edm::ParameterSet>("TH2CStripVsCpixel");
      std::string HistoName = "StripClusVsPixClus";
      GlobalCStripVsCpix = ibooker.book2D(HistoName,HistoName,
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsx"),
					       GlobalTH2Parameters.getParameter<double>("xmin"),
					       GlobalTH2Parameters.getParameter<double>("xmax"),
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsy"),
					       GlobalTH2Parameters.getParameter<double>("ymin"),
					       GlobalTH2Parameters.getParameter<double>("ymax"));
      GlobalCStripVsCpix->setAxisTitle("Strip Clusters",1);
      GlobalCStripVsCpix->setAxisTitle("Pix Clusters",2);
      
      // Absolute Bunch Crossing ***********************
      edm::ParameterSet GlobalTH1Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ABx_CSCP");
      HistoName = "AbsoluteBx_CStripVsCpixel";
      GlobalABXTH1_CSCP = ibooker.book1D(HistoName,HistoName,
						   GlobalTH1Parameters.getParameter<int32_t>("Nbinsx"),
						   GlobalTH1Parameters.getParameter<double>("xmin"),
						   GlobalTH1Parameters.getParameter<double>("xmax"));
						   GlobalABXTH1_CSCP->setAxisTitle("Absolute Bunch Crossing",1);
    }

    if (globalswitchMultiRegions){
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH2Parameters =  conf_.getParameter<edm::ParameterSet>("TH1MultiplicityRegions");
      std::string HistoName = "ClusterMultiplicityRegions";
      PixVsStripMultiplicityRegions = ibooker.book1D(HistoName,HistoName,
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinx"),
					       GlobalTH2Parameters.getParameter<double>("xmin"),
					       GlobalTH2Parameters.getParameter<double>("xmax"));
      PixVsStripMultiplicityRegions->setAxisTitle("");
      PixVsStripMultiplicityRegions->setBinLabel(1,"Main Diagonal");
      PixVsStripMultiplicityRegions->setBinLabel(2,"Strip Noise");
      PixVsStripMultiplicityRegions->setBinLabel(3,"High Strip Noise");
      PixVsStripMultiplicityRegions->setBinLabel(4,"Beam Background");
      PixVsStripMultiplicityRegions->setBinLabel(5,"No Strip Clusters");
    }

    if (globalswitchmaindiagonalposition){
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH1Parameters =  conf_.getParameter<edm::ParameterSet>("TH1MainDiagonalPosition");
      std::string HistoName = "MainDiagonal Position";
      GlobalMainDiagonalPosition = ibooker.book1D(HistoName,HistoName,
					     GlobalTH1Parameters.getParameter<int32_t>("Nbinsx"),
					     GlobalTH1Parameters.getParameter<double>("xmin"),
					     GlobalTH1Parameters.getParameter<double>("xmax"));
      GlobalMainDiagonalPosition->setAxisTitle("atan(NPix/(k*NStrip))");

      //PLOT MainDiagonalPosition_vs_BX ***************************
      edm::ParameterSet GlobalTProfParameters =  conf_.getParameter<edm::ParameterSet>("TProfMainDiagonalPosition");
      HistoName = "MainDiagonalPosition_vs_BX";
      GlobalMainDiagonalPosition_vs_BX = ibooker.bookProfile(HistoName, HistoName,
						     GlobalTProfParameters.getParameter<int32_t>("Nbinsx"),
						     GlobalTProfParameters.getParameter<double>("xmin"),
						     GlobalTProfParameters.getParameter<double>("xmax"),
						     GlobalTProfParameters.getParameter<int32_t>("Nbinsy"),
						     GlobalTProfParameters.getParameter<double>("ymin"),
						     GlobalTProfParameters.getParameter<double>("ymax"));

      GlobalMainDiagonalPosition_vs_BX->setAxisTitle("Absolute BX", 1);
      GlobalMainDiagonalPosition_vs_BX->setAxisTitle("tan^{-1}(NPix/k*NStrip))", 2);


      edm::ParameterSet GlobalTH2Parameters =  conf_.getParameter<edm::ParameterSet>("TH2MainDiagonalPosition");
      HistoName = "TH2MainDiagonalPosition_vs_BX";
      GlobalTH2MainDiagonalPosition_vs_BX = ibooker.book2D(HistoName,HistoName,
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsx"),
					       GlobalTH2Parameters.getParameter<double>("xmin"),
					       GlobalTH2Parameters.getParameter<double>("xmax"),
					       GlobalTH2Parameters.getParameter<int32_t>("Nbinsy"),
					       GlobalTH2Parameters.getParameter<double>("ymin"),
					       GlobalTH2Parameters.getParameter<double>("ymax"));
      GlobalTH2MainDiagonalPosition_vs_BX->setAxisTitle("Absolute BX",1);
      GlobalTH2MainDiagonalPosition_vs_BX->setAxisTitle("tan^{-1}(NPix/k*NStrip))",2);

      
      
    }

    // TO BE ADDED !!!
    /*
    if ( globalswitchapvcycledbxth2on or globalswitchcstripvscpix or globalswitchMultiRegions or ClusterHisto_ ) {
    ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      std::string HistoName = "BPTX rate";
      BPTXrateTrend = ibooker.bookProfile(HistoName,HistoName, LSBin, LSMin, LSMax, 0, 10000.,"");
      BPTXrateTrend->getTH1()->SetCanExtend(TH1::kAllAxes);
      BPTXrateTrend->setAxisTitle("#Lumi section",1);
      BPTXrateTrend->setAxisTitle("Number of BPTX events per LS",2);
    }
    */

    if (globalswitchstripnoise2apvcycle){
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH1Parameters =  conf_.getParameter<edm::ParameterSet>("TH1StripNoise2ApvCycle");
      std::string HistoName = "StripNoise_ApvCycle";
      StripNoise2Cycle = ibooker.book1D(HistoName,HistoName,
					     GlobalTH1Parameters.getParameter<int32_t>("Nbinsx"),
					     GlobalTH1Parameters.getParameter<double>("xmin"),
					     GlobalTH1Parameters.getParameter<double>("xmax"));
      StripNoise2Cycle->setAxisTitle("APV Cycle");
    }

    if (globalswitchstripnoise3apvcycle){
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet GlobalTH1Parameters =  conf_.getParameter<edm::ParameterSet>("TH1StripNoise3ApvCycle");
      std::string HistoName = "HighStripNoise_ApvCycle";
      StripNoise3Cycle = ibooker.book1D(HistoName,HistoName,
					     GlobalTH1Parameters.getParameter<int32_t>("Nbinsx"),
					     GlobalTH1Parameters.getParameter<double>("xmin"),
					     GlobalTH1Parameters.getParameter<double>("xmax"));
      StripNoise3Cycle->setAxisTitle("APV Cycle");
    }

    if ( globalswitchnclusvscycletimeprof2don ) {
      const char* HistoName  = "StripClusVsBXandOrbit";
      const char* HistoTitle = "Strip cluster multiplicity vs BX mod(70) and Orbit;Event 1 BX mod(70);time [Orb#]";
      edm::ParameterSet ParametersNclusVsCycleTimeProf2D = conf_.getParameter<edm::ParameterSet>("NclusVsCycleTimeProf2D");
      NclusVsCycleTimeProf2D = ibooker.bookProfile2D ( HistoName , HistoTitle ,
						       ParametersNclusVsCycleTimeProf2D.getParameter<int32_t>("Nbins"),
						       ParametersNclusVsCycleTimeProf2D.getParameter<double>("xmin"),
						       ParametersNclusVsCycleTimeProf2D.getParameter<double>("xmax"),
						       ParametersNclusVsCycleTimeProf2D.getParameter<int32_t>("Nbinsy"),
						       ParametersNclusVsCycleTimeProf2D.getParameter<double>("ymin"),
						       ParametersNclusVsCycleTimeProf2D.getParameter<double>("ymax"),
						       0 , 0 );
      if (NclusVsCycleTimeProf2D->kind() == MonitorElement::DQM_KIND_TPROFILE2D) NclusVsCycleTimeProf2D->getTH1()->SetCanExtend(TH1::kAllAxes);
    }
	if ( clusterWidth_vs_amplitude_on ) {
	  ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet ParametersClusWidthVsAmpTH2 = conf_.getParameter<edm::ParameterSet>("ClusWidthVsAmpTH2");
	  const char* HistoName  = "ClusterWidths_vs_Amplitudes";
	  const char* HistoTitle = "Cluster widths vs amplitudes;Amplitudes (integrated ADC counts);Cluster widths";
      ClusWidthVsAmpTH2 = ibooker.book2D(HistoName,HistoTitle,
                           ParametersClusWidthVsAmpTH2.getParameter<int32_t>("Nbinsx"),
                           ParametersClusWidthVsAmpTH2.getParameter<double>("xmin"),
                           ParametersClusWidthVsAmpTH2.getParameter<double>("xmax"),
                           ParametersClusWidthVsAmpTH2.getParameter<int32_t>("Nbinsy"),
                           ParametersClusWidthVsAmpTH2.getParameter<double>("ymin"),
                           ParametersClusWidthVsAmpTH2.getParameter<double>("ymax"));
	}

    if (ClusterHisto_){
      ibooker.setCurrentFolder(topFolderName_+"/MechanicalView/");
      edm::ParameterSet PixelCluster =  conf_.getParameter<edm::ParameterSet>("TH1NClusPx");
      std::string HistoName = "NumberOfClustersInPixel";
      NumberOfPixelClus = ibooker.book1D(HistoName, HistoName,
					    PixelCluster.getParameter<int32_t>("Nbinsx"),
					    PixelCluster.getParameter<double>("xmin"),
					    PixelCluster.getParameter<double>("xmax"));
      NumberOfPixelClus->setAxisTitle("# of Clusters in Pixel", 1);
      NumberOfPixelClus->setAxisTitle("Number of Events", 2);
      //
      edm::ParameterSet StripCluster =  conf_.getParameter<edm::ParameterSet>("TH1NClusStrip");
      HistoName = "NumberOfClustersInStrip";
      NumberOfStripClus = ibooker.book1D(HistoName, HistoName,
					    StripCluster.getParameter<int32_t>("Nbinsx"),
					    StripCluster.getParameter<double>("xmin"),
					    StripCluster.getParameter<double>("xmax"));
      NumberOfStripClus->setAxisTitle("# of Clusters in Strip", 1);
      NumberOfStripClus->setAxisTitle("Number of Events", 2);

      // NumberOfClustersinStrip vs BX PLOT ****************************
      edm::ParameterSet StripClusterBX =  conf_.getParameter<edm::ParameterSet>("TProfNClusStrip");
      HistoName = "NumberOfClustersInStrip_vs_BX";
      NumberOfStripClus_vs_BX = ibooker.bookProfile(HistoName, HistoName,
						    StripClusterBX.getParameter<int32_t>("Nbinsx"),
						    StripClusterBX.getParameter<double>("xmin"),
						    StripClusterBX.getParameter<double>("xmax"),
						    StripClusterBX.getParameter<int32_t>("Nbinsy"),
						    StripClusterBX.getParameter<double>("ymin"),
						    StripClusterBX.getParameter<double>("ymax"));

      NumberOfStripClus_vs_BX->setAxisTitle("Absolute BX", 1);
      NumberOfStripClus_vs_BX->setAxisTitle("# of Clusters in Strip", 2);

      // NumberOfClustersinStrip vs BX PLOT ****************************
      edm::ParameterSet PixelClusterBX =  conf_.getParameter<edm::ParameterSet>("TProfNClusPixel");
      HistoName = "NumberOfClustersInPixel_vs_BX";
      NumberOfPixelClus_vs_BX = ibooker.bookProfile(HistoName, HistoName,
						    PixelClusterBX.getParameter<int32_t>("Nbinsx"),
						    PixelClusterBX.getParameter<double>("xmin"),
						    PixelClusterBX.getParameter<double>("xmax"),
						    PixelClusterBX.getParameter<int32_t>("Nbinsy"),
						    PixelClusterBX.getParameter<double>("ymin"),
						    PixelClusterBX.getParameter<double>("ymax"));

      NumberOfPixelClus_vs_BX->setAxisTitle("Absolute BX", 1);
      NumberOfPixelClus_vs_BX->setAxisTitle("# of Clusters in Pixel", 2);
      
    }

  }//end of if
}//end of method

//--------------------------------------------------------------------------------------------

void SiStripMonitorCluster::bookHistograms(DQMStore::IBooker & ibooker, const edm::Run& run, const edm::EventSetup& es)
{
  if (show_mechanical_structure_view) {
    unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    if (m_cacheID_ != cacheID) {
      m_cacheID_ = cacheID;
     edm::LogInfo("SiStripMonitorCluster") <<"SiStripMonitorCluster::bookHistograms: "
					    << " Creating MEs for new Cabling ";

     createMEs(es , ibooker);
    }
  } else if (reset_each_run) {
    edm::LogInfo("SiStripMonitorCluster") <<"SiStripMonitorCluster::bookHistograms: "
					  << " Resetting MEs ";
    for (std::map<uint32_t, ModMEs >::const_iterator idet = ModuleMEsMap.begin() ; idet!=ModuleMEsMap.end() ; idet++) {
      ResetModuleMEs(idet->first);
    }
  }

}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Filter out events if Trigger Filtering is requested
  passBPTXfilter_     = ( iEvent.isRealData() and genTriggerEventFlagBPTXfilter_->on()     ) ? genTriggerEventFlagBPTXfilter_->accept( iEvent, iSetup)     : true;
  passPixelDCSfilter_ = ( iEvent.isRealData() and genTriggerEventFlagPixelDCSfilter_->on() ) ? genTriggerEventFlagPixelDCSfilter_->accept( iEvent, iSetup) : true;
  passStripDCSfilter_ = ( iEvent.isRealData() and genTriggerEventFlagStripDCSfilter_->on() ) ? genTriggerEventFlagStripDCSfilter_->accept( iEvent, iSetup) : true;
  //  std::cout << "passBPTXfilter_ ? " << passBPTXfilter_ << std::endl;

  // Filter out events if DCS Event if requested
  if (dcsStatus_ && !dcsStatus_->getStatus(iEvent,iSetup)) return;

  runNb   = iEvent.id().run();
  eventNb++;
  trendVar = trendVsLs_ ? iEvent.orbitNumber()/262144.0 : iEvent.orbitNumber()/11223.0; // lumisection : seconds

  int NPixClusters=0, NStripClusters=0, MultiplicityRegion=0;
  bool isPixValid=false;

  edm::ESHandle<SiStripNoises> noiseHandle;
  iSetup.get<SiStripNoisesRcd>().get(noiseHandle);

  edm::ESHandle<SiStripGain> gainHandle;
  iSetup.get<SiStripGainRcd>().get(gainHandle);

  edm::ESHandle<SiStripQuality> qualityHandle;
  iSetup.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle);

  iSetup.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);

  // get collection of DetSetVector of clusters from Event
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByToken(clusterProducerStripToken_, cluster_detsetvektor);

  //get pixel clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > cluster_detsetvektor_pix;
  iEvent.getByToken(clusterProducerPixToken_, cluster_detsetvektor_pix);

  if (!cluster_detsetvektor.isValid()) return;

  const edmNew::DetSetVector<SiStripCluster> * StrC= cluster_detsetvektor.product();
  NStripClusters= StrC->data().size();

  if (cluster_detsetvektor_pix.isValid()){
    const edmNew::DetSetVector<SiPixelCluster> * PixC= cluster_detsetvektor_pix.product();
    NPixClusters= PixC->data().size();
    isPixValid=true;
    MultiplicityRegion=FindRegion(NStripClusters,NPixClusters);

    if ( passBPTXfilter_ and passPixelDCSfilter_ and passStripDCSfilter_ ) {
      if (globalswitchcstripvscpix) GlobalCStripVsCpix->Fill(NStripClusters,NPixClusters); 
      if (globalswitchmaindiagonalposition && NStripClusters > 0) GlobalMainDiagonalPosition->Fill(atan(NPixClusters/(k0*NStripClusters)));

      if (globalswitchMultiRegions) PixVsStripMultiplicityRegions->Fill(MultiplicityRegion);
    }

    if (ClusterHisto_){
      if ( passBPTXfilter_ and passPixelDCSfilter_ )
	NumberOfPixelClus->Fill(NPixClusters);
      if ( passBPTXfilter_ and passStripDCSfilter_ )
	NumberOfStripClus->Fill(NStripClusters);
    }
  }
  // initialise # of clusters to zero
  for (std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.begin();
       iSubdet != SubDetMEsMap.end(); iSubdet++) {
    iSubdet->second.totNClusters = 0;
  }

  SiStripFolderOrganizer folder_organizer;
  bool found_layer_me = false;
  for (std::map<std::string, std::vector< uint32_t > >::const_iterator iterLayer = LayerDetMap.begin();
       iterLayer != LayerDetMap.end(); iterLayer++) {

    std::string layer_label = iterLayer->first;

    int ncluster_layer = 0;
    std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEsMap.find(layer_label);

    //get Layer MEs
    LayerMEs layer_single;
    if(iLayerME != LayerMEsMap.end()) {
       layer_single = iLayerME->second;
       found_layer_me = true;
     }

    bool found_module_me = false;
    uint16_t iDet = 0;
    std::string subdet_label = "";
    // loop over all modules in the layer
    for (std::vector< uint32_t >::const_iterator iterDets = iterLayer->second.begin() ;
	 iterDets != iterLayer->second.end() ; iterDets++) {
      iDet++;
      // detid and type of ME
      uint32_t detid = (*iterDets);

      // Get SubDet label once
      if (subdet_label.size() == 0) subdet_label = folder_organizer.getSubDetFolderAndTag(detid, tTopo).second;

      // DetId and corresponding set of MEs
      ModMEs mod_single;
      if (Mod_On_) {
	std::map<uint32_t, ModMEs >::iterator imodME = ModuleMEsMap.find(detid);
	if (imodME != ModuleMEsMap.end()) {
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
	if(clusterchtkhistomapon) tkmapclusterch->fill(detid,0.);
	if (found_layer_me && layerswitchnumclusterprofon) layer_single.LayerNumberOfClusterProfile->Fill(iDet, 0.0);
	continue; // no clusters for this detid => jump to next step of loop
      }

      //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
      //      edmNew::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'
      edmNew::DetSet<SiStripCluster> cluster_detset = (*isearch);

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

	const auto & ampls = clusterIter->amplitudes();
	// cluster position
	float cluster_position = clusterIter->barycenter();
	// start defined as nr. of first strip beloning to the cluster
	short cluster_start    = clusterIter->firstStrip();
	// width defined as nr. of strips that belong to cluster
	short cluster_width    = ampls.size();
	// add nr of strips of this cluster to total nr. of clusterized strips
	total_clusterized_strips = total_clusterized_strips + cluster_width;

	if(clusterchtkhistomapon) tkmapclusterch->fill(detid,static_cast<float>(clusterIter->charge()));

	// cluster signal and noise from the amplitudes
	float cluster_signal = 0.0;
	float cluster_noise  = 0.0;
	int nrnonzeroamplitudes = 0;
	float noise2 = 0.0;
	float noise  = 0.0;
	for(uint iamp=0; iamp<ampls.size(); iamp++){
	  if(ampls[iamp]>0){ // nonzero amplitude
	    cluster_signal += ampls[iamp];
	    if(!qualityHandle->IsStripBad(qualityRange, clusterIter->firstStrip()+iamp)){
	      noise = noiseHandle->getNoise(clusterIter->firstStrip()+iamp,detNoiseRange)/gainHandle->getStripGain(clusterIter->firstStrip()+iamp, detGainRange);
	    }
	    noise2 += noise*noise;
	    nrnonzeroamplitudes++;
	  }
	} // End loop over cluster amplitude

	if (nrnonzeroamplitudes > 0) cluster_noise = sqrt(noise2/nrnonzeroamplitudes);

	if( applyClusterQuality_ &&
	    (cluster_signal/cluster_noise < sToNLowerLimit_ ||
	     cluster_signal/cluster_noise > sToNUpperLimit_ ||
	     cluster_width < widthLowerLimit_ ||
	     cluster_width > widthUpperLimit_) ) continue;

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
          fillLayerMEs(layer_single, cluster_properties);
	  if (layerswitchclusterwidthprofon)
	    layer_single.LayerClusterWidthProfile->Fill(iDet, cluster_width);
	}
	
	if (subdetswitchcluschargeon || subdetswitchcluswidthon)
	  {
	    std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(subdet_label);
	    if(iSubdet != SubDetMEsMap.end()) 
	      {
		if (subdetswitchcluschargeon) iSubdet->second.SubDetClusterChargeTH1->Fill(cluster_signal);
		if (subdetswitchcluswidthon)  iSubdet->second.SubDetClusterWidthTH1->Fill(cluster_width);
		  }
	  }
     

    if (subdet_clusterWidth_vs_amplitude_on){
		std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(subdet_label);
		if(iSubdet != SubDetMEsMap.end()) iSubdet->second.SubDetClusWidthVsAmpTH2->Fill(cluster_signal, cluster_width); 
	}

	if ( clusterWidth_vs_amplitude_on ) {
	  ClusWidthVsAmpTH2->Fill(cluster_signal, cluster_width);
	}
	  
	if (subdetswitchtotclusprofon) {
	  std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(subdet_label);
	  std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid, tTopo);
          if(iSubdet != SubDetMEsMap.end()) iSubdet->second.SubDetNumberOfClusterPerLayerTrend->Fill(trendVar,abs(det_layer_pair.second));
        }

        if (subdetswitchtotclusprofon && (subdet_label.find("TID")!=std::string::npos || subdet_label.find("TEC")!=std::string::npos)){
	  std::pair<std::string,int32_t> det_ring_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,true);
          fillME(layer_single.LayerNumberOfClusterPerRingTrend, trendVar, abs(det_ring_pair.second));
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
	if (createTrendMEs)
	  fillME(layer_single.LayerLocalOccupancyTrend,trendVar,local_occupancy);
      }
    }

    if (subdetswitchtotclusprofon)
      fillME(layer_single.LayerNumberOfClusterTrend,trendVar,ncluster_layer);

    std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(subdet_label);
    if(iSubdet != SubDetMEsMap.end()) iSubdet->second.totNClusters += ncluster_layer;
  }

  //  EventHistory
  edm::Handle<EventWithHistory> event_history;
  iEvent.getByToken(historyProducerToken_,event_history);

  // Phase of APV
  edm::Handle<APVCyclePhaseCollection> apv_phase_collection;
  iEvent.getByToken(apvPhaseProducerToken_,apv_phase_collection);

  if (event_history.isValid()
        && !event_history.failedToGet()
        && apv_phase_collection.isValid()
        && !apv_phase_collection.failedToGet()) {


    long long dbx        = event_history->deltaBX();
    long long tbx        = event_history->absoluteBX();

    bool global_histo_filled = false;
    bool MultiplicityRegion_Vs_APVcycle_filled=false;

    // plot n 2
    if ( passBPTXfilter_ and passPixelDCSfilter_ and passStripDCSfilter_ ) {
      if (globalswitchcstripvscpix)  GlobalABXTH1_CSCP->Fill(tbx%3564);
    }
    // plot n 2

    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin();
       it != SubDetMEsMap.end(); it++) {
      std::string sdet = it->first;
      //std::string sdet = sdet_tag.substr(0,sdet_tag.find_first_of("_"));
      SubDetMEs sdetmes = it->second;

      int the_phase = APVCyclePhaseCollection::invalid;
      long long tbx_corr = tbx;
      

      if (SubDetPhasePartMap.find(sdet) != SubDetPhasePartMap.end()) the_phase = apv_phase_collection->getPhase(SubDetPhasePartMap[sdet]);
      if(the_phase==APVCyclePhaseCollection::nopartition ||
         the_phase==APVCyclePhaseCollection::multiphase ||
         the_phase==APVCyclePhaseCollection::invalid) {
	the_phase=30;
	//std::cout << " subdet " << it->first << " not valid" << " MR " << MultiplicityRegion <<std::endl;
      }
      tbx_corr  -= the_phase;

      long long dbxincycle = event_history->deltaBXinCycle(the_phase);
      //std::cout<<"dbx in cycle: " << dbxincycle << std::endl;
      if (globalswitchapvcycledbxth2on && !global_histo_filled) {
        GlobalApvCycleDBxTH2->Fill(tbx_corr%70,dbx);
	GlobalDBxTH1->Fill(dbx);
	GlobalDBxCycleTH1->Fill(dbxincycle);
        global_histo_filled = true;
      }


      // Fill MainDiagonalPosition plots ***************
      if (cluster_detsetvektor_pix.isValid()){      
	if (ClusterHisto_){
	  if ( passBPTXfilter_ and passStripDCSfilter_ ){
	    NumberOfStripClus_vs_BX->Fill(tbx%3564,NStripClusters);
	    if(passPixelDCSfilter_ ){
	      NumberOfPixelClus_vs_BX->Fill(tbx%3564,NPixClusters); 
	      if (globalswitchmaindiagonalposition && NStripClusters > 0){ 
		GlobalMainDiagonalPosition_vs_BX->Fill(tbx%3564,atan(NPixClusters/(k0*NStripClusters)));
		GlobalTH2MainDiagonalPosition_vs_BX->Fill(tbx%3564,atan(NPixClusters/(k0*NStripClusters)));
	      }
	    }
      
	  }
	}	
      }

      if (isPixValid && !MultiplicityRegion_Vs_APVcycle_filled){
	if (globalswitchstripnoise2apvcycle && MultiplicityRegion==2) {StripNoise2Cycle->Fill(tbx_corr%70);}
	if (globalswitchstripnoise3apvcycle && MultiplicityRegion==3) {StripNoise3Cycle->Fill(tbx_corr%70);}
	MultiplicityRegion_Vs_APVcycle_filled=true;
      }

      if (subdetswitchtotclusth1on)
	  sdetmes.SubDetTotClusterTH1->Fill(sdetmes.totNClusters);
      if (subdetswitchtotclusprofon)
	  sdetmes.SubDetTotClusterProf->Fill(trendVar,sdetmes.totNClusters);
      if (subdetswitchapvcycleprofon)
	sdetmes.SubDetClusterApvProf->Fill(tbx_corr%70,sdetmes.totNClusters);
      if (subdetswitchapvcycleth2on)
	sdetmes.SubDetClusterApvTH2->Fill(tbx_corr%70,sdetmes.totNClusters);
      if (subdetswitchdbxcycleprofon) {
	sdetmes.SubDetClusterDBxCycleProf->Fill(dbxincycle,sdetmes.totNClusters);
      }
      if (subdetswitchapvcycledbxprof2on)
	sdetmes.SubDetApvDBxProf2->Fill(tbx_corr%70,dbx,sdetmes.totNClusters);
    }	

    if ( globalswitchnclusvscycletimeprof2don )
      {
	long long tbx_corr = tbx;
	int the_phase = apv_phase_collection->getPhase("All");
	
	if( the_phase == APVCyclePhaseCollection::nopartition ||
	    the_phase == APVCyclePhaseCollection::multiphase ||
	    the_phase == APVCyclePhaseCollection::invalid )
	  the_phase=30;
	
	tbx_corr -= the_phase;

	NclusVsCycleTimeProf2D->Fill( tbx_corr%70 , (int)event_history->_orbit , NStripClusters );
      }
  }
}
//
// -- Reset MEs
//------------------------------------------------------------------------------
void SiStripMonitorCluster::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = ModuleMEsMap.find(idet);
  ModMEs mod_me = pos->second;

  if (moduleswitchncluson)            mod_me.NumberOfClusters->Reset();
  if (moduleswitchclusposon)          mod_me.ClusterPosition->Reset();
  if (moduleswitchclusdigiposon)      mod_me.ClusterDigiPosition->Reset();
  if (moduleswitchclusstonVsposon)    mod_me.ClusterSignalOverNoiseVsPos->Reset();
  if (moduleswitchcluswidthon)        mod_me.ClusterWidth->Reset();
  if (moduleswitchcluschargeon)       mod_me.ClusterCharge->Reset();
  if (moduleswitchclusnoiseon)        mod_me.ClusterNoise->Reset();
  if (moduleswitchclusstonon)         mod_me.ClusterSignalOverNoise->Reset();
  if (moduleswitchlocaloccupancy)     mod_me.ModuleLocalOccupancy->Reset();
  if (moduleswitchnrclusterizedstrip) mod_me.NrOfClusterizedStrips->Reset();
  if (module_clusterWidth_vs_amplitude_on) mod_me.Module_ClusWidthVsAmpTH2->Reset();
}
//
// -- Create Module Level MEs
//
void SiStripMonitorCluster::createModuleMEs(ModMEs& mod_single, uint32_t detid , DQMStore::IBooker & ibooker) {

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  std::string hid;

  //nr. of clusters per module
  if(moduleswitchncluson) {
    hid = hidmanager.createHistoId("NumberOfClusters","det",detid);
    mod_single.NumberOfClusters = bookME1D("TH1nClusters", hid.c_str() , ibooker);
    ibooker.tag(mod_single.NumberOfClusters, detid);
    mod_single.NumberOfClusters->setAxisTitle("number of clusters in one detector module");
    mod_single.NumberOfClusters->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }

  //ClusterPosition
  if(moduleswitchclusposon) {
    short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs
    hid = hidmanager.createHistoId("ClusterPosition","det",detid);
    mod_single.ClusterPosition = ibooker.book1D(hid, hid, total_nr_strips, 0.5, total_nr_strips+0.5);
    ibooker.tag(mod_single.ClusterPosition, detid);
    mod_single.ClusterPosition->setAxisTitle("cluster position [strip number +0.5]");
  }

  //ClusterDigiPosition
  if(moduleswitchclusdigiposon) {
    short total_nr_strips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128; // get correct # of avp pairs
    hid = hidmanager.createHistoId("ClusterDigiPosition","det",detid);
    mod_single.ClusterDigiPosition = ibooker.book1D(hid, hid, total_nr_strips, 0.5, total_nr_strips+0.5);
    ibooker.tag(mod_single.ClusterDigiPosition, detid);
    mod_single.ClusterDigiPosition->setAxisTitle("digi in cluster position [strip number +0.5]");
  }

  //ClusterWidth
  if(moduleswitchcluswidthon) {
    hid = hidmanager.createHistoId("ClusterWidth","det",detid);
    mod_single.ClusterWidth = bookME1D("TH1ClusterWidth", hid.c_str() , ibooker);
    ibooker.tag(mod_single.ClusterWidth, detid);
    mod_single.ClusterWidth->setAxisTitle("cluster width [nr strips]");
  }

  //ClusterCharge
  if(moduleswitchcluschargeon) {
    hid = hidmanager.createHistoId("ClusterCharge","det",detid);
    mod_single.ClusterCharge = bookME1D("TH1ClusterCharge", hid.c_str() , ibooker );
    ibooker.tag(mod_single.ClusterCharge, detid);
    mod_single.ClusterCharge->setAxisTitle("cluster charge [ADC]");
  }

  //ClusterNoise
  if(moduleswitchclusnoiseon) {
    hid = hidmanager.createHistoId("ClusterNoise","det",detid);
    mod_single.ClusterNoise = bookME1D("TH1ClusterNoise", hid.c_str() , ibooker );
    ibooker.tag(mod_single.ClusterNoise, detid);
    mod_single.ClusterNoise->setAxisTitle("cluster noise");
  }

  //ClusterSignalOverNoise
  if(moduleswitchclusstonon) {
    hid = hidmanager.createHistoId("ClusterSignalOverNoise","det",detid);
    mod_single.ClusterSignalOverNoise = bookME1D("TH1ClusterStoN", hid.c_str() , ibooker);
    ibooker.tag(mod_single.ClusterSignalOverNoise, detid);
    mod_single.ClusterSignalOverNoise->setAxisTitle("ratio of signal to noise for each cluster");
  }

  //ClusterSignalOverNoiseVsPos
  if(moduleswitchclusstonVsposon) {
    hid = hidmanager.createHistoId("ClusterSignalOverNoiseVsPos","det",detid);
    Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoNVsPos");
    mod_single.ClusterSignalOverNoiseVsPos= ibooker.bookProfile(hid.c_str(),hid.c_str(),
								 Parameters.getParameter<int32_t>("Nbinx"),
								 Parameters.getParameter<double>("xmin"),
								 Parameters.getParameter<double>("xmax"),
								 Parameters.getParameter<int32_t>("Nbiny"),
								 Parameters.getParameter<double>("ymin"),
								 Parameters.getParameter<double>("ymax")
								 );
    ibooker.tag(mod_single.ClusterSignalOverNoiseVsPos, detid);
    mod_single.ClusterSignalOverNoiseVsPos->setAxisTitle("pos");
  }

  //ModuleLocalOccupancy
  if (moduleswitchlocaloccupancy) {
    hid = hidmanager.createHistoId("ClusterLocalOccupancy","det",detid);
    mod_single.ModuleLocalOccupancy = bookME1D("TH1ModuleLocalOccupancy", hid.c_str() , ibooker);
    ibooker.tag(mod_single.ModuleLocalOccupancy, detid);
    mod_single.ModuleLocalOccupancy->setAxisTitle("module local occupancy [% of clusterized strips]");
  }

  //NrOfClusterizedStrips
  if (moduleswitchnrclusterizedstrip) {
    hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",detid);
    mod_single.NrOfClusterizedStrips = bookME1D("TH1NrOfClusterizedStrips", hid.c_str() , ibooker );
    ibooker.tag(mod_single.NrOfClusterizedStrips, detid);
    mod_single.NrOfClusterizedStrips->setAxisTitle("number of clusterized strips");
  }

  if (module_clusterWidth_vs_amplitude_on){
    hid = hidmanager.createHistoId("ClusterWidths_vs_Amplitudes","det",detid);
    Parameters =  conf_.getParameter<edm::ParameterSet>("ClusWidthVsAmpTH2");
    int32_t Nbinsx = Parameters.getParameter<int32_t>("Nbinsx");
    Nbinsx = Nbinsx/2;	//Without this "rebinning" the job is killed on lxplus. We think it's due to the high memory needed to create all those histograms.
    mod_single.Module_ClusWidthVsAmpTH2 = ibooker.book2D(hid.c_str() , hid.c_str(),
            						Nbinsx,
            						Parameters.getParameter<double>("xmin"),
            						Parameters.getParameter<double>("xmax"),
            						Parameters.getParameter<int32_t>("Nbinsy"),
            						Parameters.getParameter<double>("ymin"),
            						Parameters.getParameter<double>("ymax")
            						);
    ibooker.tag(mod_single.Module_ClusWidthVsAmpTH2, detid);
    mod_single.Module_ClusWidthVsAmpTH2->setAxisTitle("Amplitudes (integrated ADC counts)",1);
    mod_single.Module_ClusWidthVsAmpTH2->setAxisTitle("Cluster widths",2);
  }

}
//
// -- Create Module Level MEs
//
void SiStripMonitorCluster::createLayerMEs(std::string label, int ndets , DQMStore::IBooker & ibooker) {

  SiStripHistoId hidmanager;

  LayerMEs layerMEs;
  layerMEs.LayerClusterStoN = 0;
  layerMEs.LayerClusterStoNTrend = 0;
  layerMEs.LayerClusterCharge = 0;
  layerMEs.LayerClusterChargeTrend = 0;
  layerMEs.LayerClusterNoise = 0;
  layerMEs.LayerClusterNoiseTrend = 0;
  layerMEs.LayerClusterWidth = 0;
  layerMEs.LayerClusterWidthTrend = 0;
  layerMEs.LayerLocalOccupancy = 0;
  layerMEs.LayerLocalOccupancyTrend = 0;
  layerMEs.LayerNumberOfClusterProfile = 0;
  layerMEs.LayerNumberOfClusterTrend = 0;
  layerMEs.LayerNumberOfClusterPerRingTrend = 0;
  layerMEs.LayerClusterWidthProfile = 0;
  layerMEs.LayerClusWidthVsAmpTH2 = 0;
  layerMEs.LayerClusterPosition = 0;

  //Cluster Width
  if(layerswitchcluswidthon) {
    layerMEs.LayerClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoLayer("Summary_ClusterWidth","layer",label,"").c_str() , ibooker );
    if (createTrendMEs)
      layerMEs.LayerClusterWidthTrend=bookMETrend(hidmanager.createHistoLayer("Trend_ClusterWidth","layer",label,"").c_str() , ibooker );
  }

  //Cluster Noise
  if(layerswitchclusnoiseon) {
    layerMEs.LayerClusterNoise=bookME1D("TH1ClusterNoise", hidmanager.createHistoLayer("Summary_ClusterNoise","layer",label,"").c_str() , ibooker );
    if (createTrendMEs)
      layerMEs.LayerClusterNoiseTrend=bookMETrend(hidmanager.createHistoLayer("Trend_ClusterNoise","layer",label,"").c_str() , ibooker);
  }

  //Cluster Charge
  if(layerswitchcluschargeon) {
    layerMEs.LayerClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoLayer("Summary_ClusterCharge","layer",label,"").c_str() , ibooker );
    if (createTrendMEs)
      layerMEs.LayerClusterChargeTrend=bookMETrend(hidmanager.createHistoLayer("Trend_ClusterCharge","layer",label,"").c_str(),ibooker);
  }

  //Cluster StoN
  if(layerswitchclusstonon) {
    layerMEs.LayerClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoLayer("Summary_ClusterSignalOverNoise","layer",label,"").c_str() , ibooker );
    if (createTrendMEs)
      layerMEs.LayerClusterStoNTrend=bookMETrend(hidmanager.createHistoLayer("Trend_ClusterSignalOverNoise","layer",label,"").c_str(),ibooker);
  }

  //Cluster Occupancy
  if(layerswitchlocaloccupancy) {
    layerMEs.LayerLocalOccupancy=bookME1D("TH1ModuleLocalOccupancy", hidmanager.createHistoLayer("Summary_ClusterLocalOccupancy","layer",label,"").c_str() , ibooker );
    if (createTrendMEs)
      layerMEs.LayerLocalOccupancyTrend=bookMETrend(hidmanager.createHistoLayer("Trend_ClusterLocalOccupancy","layer",label,"").c_str(),ibooker);
  }

  // # of Cluster Profile
  if(layerswitchnumclusterprofon) {
    std::string hid = hidmanager.createHistoLayer("NumberOfClusterProfile","layer",label,"");
    layerMEs.LayerNumberOfClusterProfile = ibooker.bookProfile(hid, hid, ndets, 0.5, ndets+0.5,21, -0.5, 20.5);
  }

  // # of Cluster Trend
  if (subdetswitchtotclusprofon){
    layerMEs.LayerNumberOfClusterTrend = bookMETrend(hidmanager.createHistoLayer("NumberOfClusterTrend","layer",label,"").c_str(),ibooker);

    if (label.find("TID")!=std::string::npos || label.find("TEC")!=std::string::npos){
      edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("NumberOfClusterPerRingVsTrendVarTH2");
      layerMEs.LayerNumberOfClusterPerRingTrend = bookME2D("NumberOfClusterPerRingVsTrendVarTH2", hidmanager.createHistoLayer("NumberOfClusterPerRing_vs_TrendVar","layer",label,"").c_str() , ibooker );
    }
  }

  // Cluster Width Profile
  if(layerswitchclusterwidthprofon) {
    std::string hid = hidmanager.createHistoLayer("ClusterWidthProfile","layer",label,"");
    layerMEs.LayerClusterWidthProfile = ibooker.bookProfile(hid, hid, ndets, 0.5, ndets+0.5, 20, -0.5, 19.5);
  }

  if(layer_clusterWidth_vs_amplitude_on) {
    layerMEs.LayerClusWidthVsAmpTH2=bookME2D("ClusWidthVsAmpTH2", hidmanager.createHistoLayer("ClusterWidths_vs_Amplitudes","layer",label,"").c_str() , ibooker );
  }

  // Cluster Position
  if (layerswitchclusposon)  {
    std::string hid = hidmanager.createHistoLayer("ClusterPosition","layer",label,"");
    layerMEs.LayerClusterPosition=ibooker.book1D(hid, hid, 128*6, 0.5, 128*6+0.5);
  }

  LayerMEsMap[label]=layerMEs;
}
//
// -- Create SubDetector MEs
//
void SiStripMonitorCluster::createSubDetMEs(std::string label , DQMStore::IBooker & ibooker ) {

  SubDetMEs subdetMEs;
  subdetMEs.totNClusters              = 0;
  subdetMEs.SubDetTotClusterTH1       = 0;
  subdetMEs.SubDetTotClusterProf      = 0;
  subdetMEs.SubDetClusterApvProf      = 0;
  subdetMEs.SubDetClusterApvTH2       = 0;
  subdetMEs.SubDetClusterDBxCycleProf = 0;
  subdetMEs.SubDetApvDBxProf2         = 0;
  subdetMEs.SubDetClusterChargeTH1    = 0;
  subdetMEs.SubDetClusterWidthTH1     = 0;
  subdetMEs.SubDetClusWidthVsAmpTH2	  = 0;
  subdetMEs.SubDetNumberOfClusterPerLayerTrend    = 0;

  std::string HistoName;
  // cluster charge
  if (subdetswitchcluschargeon){
    HistoName = "ClusterCharge__" + label;
    subdetMEs.SubDetClusterChargeTH1 = bookME1D("TH1ClusterCharge",HistoName.c_str() , ibooker);
    subdetMEs.SubDetClusterChargeTH1->setAxisTitle("Cluster charge [ADC counts]");
    subdetMEs.SubDetClusterChargeTH1->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
  // cluster width
  if (subdetswitchcluswidthon){
    HistoName = "ClusterWidth__" + label;
    subdetMEs.SubDetClusterWidthTH1 = bookME1D("TH1ClusterWidth",HistoName.c_str() , ibooker);
    subdetMEs.SubDetClusterWidthTH1->setAxisTitle("Cluster width [strips]");
    subdetMEs.SubDetClusterWidthTH1->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
  // Total Number of Cluster - 1D
  if (subdetswitchtotclusth1on){
    HistoName = "TotalNumberOfCluster__" + label;
    subdetMEs.SubDetTotClusterTH1 = bookME1D("TH1TotalNumberOfClusters",HistoName.c_str() , ibooker);
    subdetMEs.SubDetTotClusterTH1->setAxisTitle("Total number of clusters in subdetector");
    subdetMEs.SubDetTotClusterTH1->getTH1()->StatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
  // Total Number of Cluster vs Time - Profile
  if (subdetswitchtotclusprofon){
    edm::ParameterSet Parameters = trendVsLs_ ? conf_.getParameter<edm::ParameterSet>("TrendingLS") : conf_.getParameter<edm::ParameterSet>("Trending");
    HistoName = "TotalNumberOfClusterProfile__" + label;
    subdetMEs.SubDetTotClusterProf = ibooker.bookProfile(HistoName,HistoName,
							 Parameters.getParameter<int32_t>("Nbins"),
							 Parameters.getParameter<double>("xmin"),
							 Parameters.getParameter<double>("xmax"),
							 0 , 0 , "" );
    subdetMEs.SubDetTotClusterProf->setAxisTitle(Parameters.getParameter<std::string>("xaxis"),1);
    if (subdetMEs.SubDetTotClusterProf->kind() == MonitorElement::DQM_KIND_TPROFILE) subdetMEs.SubDetTotClusterProf->getTH1()->SetCanExtend(TH1::kAllAxes);
  
    HistoName = "TotalNumberOfClusterPerLayer__" + label;
    subdetMEs.SubDetNumberOfClusterPerLayerTrend = bookME2D("NumberOfClusterPerLayerTrendVarTH2", HistoName.c_str() , ibooker );
    subdetMEs.SubDetNumberOfClusterPerLayerTrend->setAxisTitle("Lumisection",1);
    subdetMEs.SubDetNumberOfClusterPerLayerTrend->setAxisTitle("Layer Number",2);
  }

  // Total Number of Cluster vs APV cycle - Profile
  if(subdetswitchapvcycleprofon){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TProfClustersApvCycle");
    HistoName = "Cluster_vs_ApvCycle__" + label;
    subdetMEs.SubDetClusterApvProf=ibooker.bookProfile(HistoName,HistoName,
							  Parameters.getParameter<int32_t>("Nbins"),
							  Parameters.getParameter<double>("xmin"),
							  Parameters.getParameter<double>("xmax"),
							  200, //that parameter should not be there !?
							  Parameters.getParameter<double>("ymin"),
							  Parameters.getParameter<double>("ymax"),
							  "" );
    subdetMEs.SubDetClusterApvProf->setAxisTitle("Apv Cycle (Corrected Absolute Bx % 70)",1);
  }

  // Total Number of Clusters vs ApvCycle - 2D
  if(subdetswitchapvcycleth2on){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TH2ClustersApvCycle");
    HistoName = "Cluster_vs_ApvCycle_2D__" + label;
    // Adjusting the scale for 2D histogram
    double h2ymax = 9999.0;
    double yfact = Parameters.getParameter<double>("yfactor");
    if(label.find("TIB") != std::string::npos) h2ymax = (6984.*256.)*yfact;
    else if (label.find("TID") != std::string::npos) h2ymax = (2208.*256.)*yfact;
    else if (label.find("TOB") != std::string::npos) h2ymax = (12906.*256.)*yfact;
    else if (label.find("TEC") != std::string::npos) h2ymax = (7552.*2.*256.)*yfact;

    subdetMEs.SubDetClusterApvTH2=ibooker.book2D(HistoName,HistoName,
						    Parameters.getParameter<int32_t>("Nbinsx"),
						    Parameters.getParameter<double>("xmin"),
						    Parameters.getParameter<double>("xmax"),
						    Parameters.getParameter<int32_t>("Nbinsy"),
						    Parameters.getParameter<double>("ymin"),
						    h2ymax);
    subdetMEs.SubDetClusterApvTH2->setAxisTitle("Apv Cycle (Corrected Absolute Bx % 70))",1);
    subdetMEs.SubDetClusterApvTH2->setAxisTitle("Total # of Clusters",2);

  }

  // Cluster widths vs amplitudes - 2D
  if(subdet_clusterWidth_vs_amplitude_on){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("ClusWidthVsAmpTH2");
    HistoName = "ClusterWidths_vs_Amplitudes__" + label;
    subdetMEs.SubDetClusWidthVsAmpTH2=ibooker.book2D(HistoName,HistoName,
						    Parameters.getParameter<int32_t>("Nbinsx"),
						    Parameters.getParameter<double>("xmin"),
						    Parameters.getParameter<double>("xmax"),
						    Parameters.getParameter<int32_t>("Nbinsy"),
						    Parameters.getParameter<double>("ymin"),
						    Parameters.getParameter<double>("ymax"));
    subdetMEs.SubDetClusWidthVsAmpTH2->setAxisTitle("Amplitudes (integrated ADC counts)",1);
    subdetMEs.SubDetClusWidthVsAmpTH2->setAxisTitle("Cluster widths",2);

  }

  
  
  // Total Number of Cluster vs DeltaBxCycle - Profile
  if(subdetswitchdbxcycleprofon){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TProfClustersVsDBxCycle");
    HistoName = "Cluster_vs_DeltaBxCycle__" + label;
    subdetMEs.SubDetClusterDBxCycleProf = ibooker.bookProfile(HistoName,HistoName,
							      Parameters.getParameter<int32_t>("Nbins"),
							      Parameters.getParameter<double>("xmin"),
							      Parameters.getParameter<double>("xmax"),
							      200, //that parameter should not be there !?
							      Parameters.getParameter<double>("ymin"),
							      Parameters.getParameter<double>("ymax"),
							      "" );
    subdetMEs.SubDetClusterDBxCycleProf->setAxisTitle("Delta Bunch Crossing Cycle",1);
  }
  // DeltaBx vs ApvCycle - 2DProfile
  if(subdetswitchapvcycledbxprof2on){
    edm::ParameterSet Parameters =  conf_.getParameter<edm::ParameterSet>("TProf2ApvCycleVsDBx");
    HistoName = "DeltaBx_vs_ApvCycle__" + label;
    subdetMEs.SubDetApvDBxProf2 = ibooker.bookProfile2D(HistoName,HistoName,
							   Parameters.getParameter<int32_t>("Nbinsx"),
							   Parameters.getParameter<double>("xmin"),
							   Parameters.getParameter<double>("xmax"),
							   Parameters.getParameter<int32_t>("Nbinsy"),
							   Parameters.getParameter<double>("ymin"),
							   Parameters.getParameter<double>("ymax"),
							   Parameters.getParameter<double>("zmin"),
							   Parameters.getParameter<double>("zmax"),
							   "" );
    subdetMEs.SubDetApvDBxProf2->setAxisTitle("APV Cycle (Corrected Absolute Bx % 70)",1);
    subdetMEs.SubDetApvDBxProf2->setAxisTitle("Delta Bunch Crossing Cycle",2);
  }
  SubDetMEsMap[label]=subdetMEs;
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

  if(module_clusterWidth_vs_amplitude_on) (mod_mes.Module_ClusWidthVsAmpTH2)->Fill(cluster.charge, cluster.width);
}
//
// -- Fill Layer Level MEs
//
void SiStripMonitorCluster::fillLayerMEs(LayerMEs& layerMEs, ClusterProperties& cluster) {

  if(layerswitchclusstonon) {
    fillME(layerMEs.LayerClusterStoN  ,cluster.charge/cluster.noise);
    if (createTrendMEs) {
	fillME(layerMEs.LayerClusterStoNTrend,trendVar,cluster.charge/cluster.noise);
      }
  }

  if(layerswitchcluschargeon) {
    fillME(layerMEs.LayerClusterCharge,cluster.charge);
    if (createTrendMEs) {
	fillME(layerMEs.LayerClusterChargeTrend,trendVar,cluster.charge);
      }
  }

  if(layerswitchclusnoiseon) {
    fillME(layerMEs.LayerClusterNoise ,cluster.noise);
    if (createTrendMEs) {
	fillME(layerMEs.LayerClusterNoiseTrend,trendVar,cluster.noise);
      }
  }

  if(layerswitchcluswidthon) {
    fillME(layerMEs.LayerClusterWidth ,cluster.width);
    if (createTrendMEs) {
	fillME(layerMEs.LayerClusterWidthTrend,trendVar,cluster.width);
      }
  }

  if( layer_clusterWidth_vs_amplitude_on ) {
	fillME(layerMEs.LayerClusWidthVsAmpTH2, cluster.charge, cluster.width);
  }

  if (layerswitchclusposon)  {
    fillME(layerMEs.LayerClusterPosition,cluster.position);
  }
}
//------------------------------------------------------------------------------------------
MonitorElement* SiStripMonitorCluster::bookMETrend(const char* HistoName , DQMStore::IBooker & ibooker)
{
  edm::ParameterSet ParametersTrend = trendVsLs_ ? conf_.getParameter<edm::ParameterSet>("TrendingLS") : conf_.getParameter<edm::ParameterSet>("Trending");
  MonitorElement* me = ibooker.bookProfile(HistoName,HistoName,
					   ParametersTrend.getParameter<int32_t>("Nbins"),
					   ParametersTrend.getParameter<double>("xmin"),
					   ParametersTrend.getParameter<double>("xmax"),
					   0 , 0 , "" );
  if(!me) return me;
  me->setAxisTitle(ParametersTrend.getParameter<std::string>("xaxis"),1);
  if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) me->getTH1()->SetCanExtend(TH1::kAllAxes);
  return me;
}

//------------------------------------------------------------------------------------------
MonitorElement* SiStripMonitorCluster::bookME1D(const char* ParameterSetLabel, const char* HistoName , DQMStore::IBooker & ibooker)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,HistoName,
			   Parameters.getParameter<int32_t>("Nbinx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax")
			   );
}

//------------------------------------------------------------------------------------------
MonitorElement* SiStripMonitorCluster::bookME2D(const char* ParameterSetLabel, const char* HistoName , DQMStore::IBooker & ibooker)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book2D(HistoName,HistoName,
			   Parameters.getParameter<int32_t>("Nbinsx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax"),
			   Parameters.getParameter<int32_t>("Nbinsy"),
			   Parameters.getParameter<double>("ymin"),
			   Parameters.getParameter<double>("ymax")
			   );
}


int SiStripMonitorCluster::FindRegion(int nstrip,int npix){

  double kplus= k0*(1+dk0/100);
  double kminus=k0*(1-dk0/100);
  int region=0;

  if (nstrip!=0 && npix >= (nstrip*kminus-q0) && npix <=(nstrip*kplus+q0)) region=1;
  else if (nstrip!=0 && npix < (nstrip*kminus-q0) &&  nstrip <= maxClus) region=2;
  else if (nstrip!=0 && npix < (nstrip*kminus-q0) &&  nstrip > maxClus) region=3;
  else if (nstrip!=0 && npix > (nstrip*kplus+q0)) region=4;
  else if (npix > minPix && nstrip==0) region=5;
  return region;

}
