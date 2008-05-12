// -*- C++ -*-
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc
*/
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
// $Id: SiStripMonitorCluster.cc,v 1.40 2008/04/29 15:00:43 dutta Exp $
#include <vector>
#include <numeric>
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//--------------------------------------------------------------------------------------------
SiStripMonitorCluster::SiStripMonitorCluster(const edm::ParameterSet& iConfig) : dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(false), reset_each_run(false), fill_signal_noise (false) {} 
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
    for (std::map<uint32_t, ModMEs >::const_iterator idet = ClusterMEs.begin() ; idet!=ClusterMEs.end() ; idet++) {
      ResetModuleMEs(idet->first);
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::endRun(const edm::Run&, const edm::EventSetup&){
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file
   show_mechanical_structure_view = conf_.getParameter<bool>("ShowMechanicalStructureView");
   show_readout_view = conf_.getParameter<bool>("ShowReadoutView");
   show_control_view = conf_.getParameter<bool>("ShowControlView");
   select_all_detectors = conf_.getParameter<bool>("SelectAllDetectors");
   reset_each_run = conf_.getParameter<bool>("ResetMEsEachRun");
   fill_signal_noise = conf_.getParameter<bool>("FillSignalNoiseHistos");
   
   edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowMechanicalStructureView = "<<show_mechanical_structure_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowReadoutView = "<<show_readout_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowControlView = "<<show_control_view;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"SelectAllDetectors = "<<select_all_detectors;
   edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ResetMEsEachRun = "<<reset_each_run;
}
//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::createMEs(const edm::EventSetup& es){
  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    es.get<SiStripDetCablingRcd>().get(tkmechstruct);
    
    // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
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
    
    folder_organizer.setSiStripFolder();

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
    for(std::vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs modSingle;
      std::string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
      if (reset_each_run) ResetModuleMEs(*detid_iterator);
      //nr. of clusters per module
      hid = hidmanager.createHistoId("NumberOfClusters","det",*detid_iterator);
      modSingle.NumberOfClusters = dqmStore_->book1D(hid, hid, 5,-0.5,4.5); dqmStore_->tag(modSingle.NumberOfClusters, *detid_iterator);
      modSingle.NumberOfClusters->setAxisTitle("number of clusters in one detector module");
      //ClusterPosition
      hid = hidmanager.createHistoId("ClusterPosition","det",*detid_iterator);
      modSingle.ClusterPosition = dqmStore_->book1D(hid, hid, 768,-0.5,767.5); dqmStore_->tag(modSingle.ClusterPosition, *detid_iterator); // 6 APVs -> 768 strips
      modSingle.ClusterPosition->setAxisTitle("cluster position [strip number +0.5]");
      //ClusterWidth
      hid = hidmanager.createHistoId("ClusterWidth","det",*detid_iterator);
      modSingle.ClusterWidth = dqmStore_->book1D(hid, hid, 20,-0.5,19.5); dqmStore_->tag(modSingle.ClusterWidth, *detid_iterator);
      modSingle.ClusterWidth->setAxisTitle("cluster width [nr strips]");
      //ClusterCharge
      hid = hidmanager.createHistoId("ClusterCharge","det",*detid_iterator);
      modSingle.ClusterCharge = dqmStore_->book1D(hid, hid, 100,0.,500.); dqmStore_->tag(modSingle.ClusterCharge, *detid_iterator);
      modSingle.ClusterCharge->setAxisTitle("cluster charge [ADC]");
      //ClusterNoise
      hid = hidmanager.createHistoId("ClusterNoise","det",*detid_iterator);
      modSingle.ClusterNoise = dqmStore_->book1D(hid, hid, 20,0.,10.); dqmStore_->tag(modSingle.ClusterNoise, *detid_iterator);
      modSingle.ClusterNoise->setAxisTitle("cluster noise");
      //ClusterSignalOverNoise
      hid = hidmanager.createHistoId("ClusterSignalOverNoise","det",*detid_iterator);
      modSingle.ClusterSignalOverNoise = dqmStore_->book1D(hid, hid, 60,0.,200.); dqmStore_->tag(modSingle.ClusterSignalOverNoise, *detid_iterator);
      modSingle.ClusterSignalOverNoise->setAxisTitle("ratio of signal to noise for each cluster");
      //ModuleLocalOccupancy
      hid = hidmanager.createHistoId("ModuleLocalOccupancy","det",*detid_iterator);
      // occupancy goes from 0 to 1, probably not over some limit value (here 0.1)
      modSingle.ModuleLocalOccupancy = dqmStore_->book1D(hid, hid, 20,-0.005,0.05); dqmStore_->tag(modSingle.ModuleLocalOccupancy, *detid_iterator);
      modSingle.ModuleLocalOccupancy->setAxisTitle("module local occupancy [% of clusterized strips]");
      //NrOfClusterizedStrips
      hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",*detid_iterator);
      modSingle.NrOfClusterizedStrips = dqmStore_->book1D(hid, hid, 10,-0.5,9.5); dqmStore_->tag(modSingle.NrOfClusterizedStrips, *detid_iterator);
      modSingle.NrOfClusterizedStrips->setAxisTitle("number of clusterized strips");
      // append to ClusterMEs
      ClusterMEs.insert( std::make_pair(*detid_iterator, modSingle));
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

 using namespace edm;
  edm::ESHandle<SiStripNoises> noiseHandle;
  iSetup.get<SiStripNoisesRcd>().get(noiseHandle);

  edm::ESHandle<SiStripGain> gainHandle;
  iSetup.get<SiStripGainRcd>().get(gainHandle);

  
  std::string quality_label  = conf_.getParameter<std::string>("StripQualityLabel");
  edm::ESHandle<SiStripQuality> qualityHandle;
  iSetup.get<SiStripQualityRcd>().get(quality_label,qualityHandle);


  // retrieve producer name of input StripClusterCollection
  std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
  // get collection of DetSetVector of clusters from Event
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByLabel(clusterProducer, cluster_detsetvektor);
  if (!cluster_detsetvektor.isValid()) return;
  // loop over MEs. Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
  for (std::map<uint32_t, ModMEs>::const_iterator iterMEs = ClusterMEs.begin() ; iterMEs!=ClusterMEs.end() ; iterMEs++) {
    uint32_t detid = iterMEs->first;  ModMEs modSingle = iterMEs->second;
    // get from DetSetVector the DetSet of clusters belonging to one detid - first make sure there exists clusters with this id
    edmNew::DetSetVector<SiStripCluster>::const_iterator isearch = cluster_detsetvektor->find(detid); // search  clusters of detid
    if(isearch==cluster_detsetvektor->end()){
      if(modSingle.NumberOfClusters != NULL){
        (modSingle.NumberOfClusters)->Fill(0.,1.); // no clusters for this detector module, so fill histogram with 0
      }
      continue; // no clusters for this detid => jump to next step of loop
    }
    //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
    edmNew::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'

    if(modSingle.NumberOfClusters != NULL){ // nr. of clusters per module
      (modSingle.NumberOfClusters)->Fill(static_cast<float>(cluster_detset.size()),1.);
    }


    short total_clusterized_strips = 0;
    //
    float clusterNoise = 0.;
    float clusterNoise2 = 0;
    int nrnonzeroamplitudes = 0;

    SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detid);
    SiStripApvGain::Range detGainRange =  gainHandle->getRange(detid); 
    SiStripQuality::Range qualityRange = qualityHandle->getRange(detid);

    for(edmNew::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.begin(); clusterIter!= cluster_detset.end(); clusterIter++){
      
      if(modSingle.ClusterPosition != NULL){ // position of cluster
	(modSingle.ClusterPosition)->Fill(clusterIter->barycenter(),1.);
      }
    
      if(modSingle.ClusterWidth != NULL){ // width of cluster, calculate yourself, no method for getting it
	const std::vector<uint8_t>& ampls = clusterIter->amplitudes();
        short local_size = ampls.size(); // width defined as nr. of strips that belong to cluster
        total_clusterized_strips = total_clusterized_strips + local_size; // add nr of strips of this cluster to total nr. of clusterized strips
        (modSingle.ClusterWidth)->Fill(static_cast<float>(local_size),1.);
      }
    
      if( fill_signal_noise && modSingle.ClusterSignalOverNoise){
	const std::vector<uint8_t>& ampls = clusterIter->amplitudes();
	float clusterSignal = 0;
        float noise;
        for(uint iamp=0; iamp<ampls.size(); iamp++){
          if(ampls[iamp]>0){ // nonzero amplitude
	    clusterSignal += ampls[iamp];
	    try{
	      if(!qualityHandle->IsStripBad(qualityRange, clusterIter->firstStrip()+iamp)){
	        noise = noiseHandle->getNoise(clusterIter->firstStrip()+iamp,detNoiseRange)/gainHandle->getStripGain(clusterIter->firstStrip()+iamp, detGainRange);
	      }
            }catch(cms::Exception& e){
              edm::LogError("SiStripTkDQM|SiStripMonitorCluster|DB")<<" cms::Exception:  detid="<<detid<<" firstStrip="<<clusterIter->firstStrip()<<" iamp="<<iamp<<e.what();
            }
            clusterNoise2 += noise*noise;
            nrnonzeroamplitudes++;
          }
        }
        clusterNoise = sqrt(clusterNoise2/nrnonzeroamplitudes);
        if(modSingle.ClusterNoise) (modSingle.ClusterNoise)->Fill(clusterNoise,1.);
        if(modSingle.ClusterSignalOverNoise) (modSingle.ClusterSignalOverNoise)->Fill(clusterSignal/clusterNoise,1.);
      }
      
      //
      if(modSingle.ClusterCharge != NULL){ // charge of cluster
      	const std::vector<uint8_t>& ampls = clusterIter->amplitudes();
	short local_charge = 0;
        for(std::vector<uint8_t>::const_iterator iampls = ampls.begin(); iampls<ampls.end(); iampls++){
          local_charge += *iampls;
        }
        (modSingle.ClusterCharge)->Fill(static_cast<float>(local_charge),1.);
      }

    } // end loop on clusters for the given detid

    if(modSingle.NrOfClusterizedStrips != NULL){ // nr of clusterized strips
      modSingle.NrOfClusterizedStrips->Fill(static_cast<float>(total_clusterized_strips),1.);
    }

    short total_nr_strips = 6 * 128; // assume 6 APVs per detector for the moment. later ask FedCabling object
    float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
    if(modSingle.ModuleLocalOccupancy != NULL){ // nr of clusterized strips
      modSingle.ModuleLocalOccupancy->Fill(local_occupancy,1.);
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    std::ofstream monitor_summary("monitor_cluster_summary.txt");
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    monitor_summary<<"SiStripMonitorCluster::endJob ClusterMEs.size()="<<ClusterMEs.size()<<std::endl;
    for(std::map<uint32_t, ModMEs>::const_iterator idet = ClusterMEs.begin(); idet!= ClusterMEs.end(); idet++ ){
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"      ++++++detid  "<<idet->first<<std::endl<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ NumberOfClusters "<<(idet->second).NumberOfClusters->getEntries()<<" "<<(idet->second).NumberOfClusters->getMean()<<" "<<(idet->second).NumberOfClusters->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterPosition "<<(idet->second).ClusterPosition->getEntries()<<" "<<(idet->second).ClusterPosition->getMean()<<" "<<(idet->second).ClusterPosition->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterWidth "<<(idet->second).ClusterWidth->getEntries()<<" "<<(idet->second).ClusterWidth->getMean()<<" "<<(idet->second).ClusterWidth->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterCharge "<<(idet->second).ClusterCharge->getEntries()<<" "<<(idet->second).ClusterCharge->getMean()<<" "<<(idet->second).ClusterCharge->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterNoise "<<(idet->second).ClusterNoise->getEntries()<<" "<<(idet->second).ClusterNoise->getMean()<<" "<<(idet->second).ClusterNoise->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterSignalOverNoise "<<(idet->second).ClusterSignalOverNoise->getEntries()<<" "<<(idet->second).ClusterSignalOverNoise->getMean()<<" "<<(idet->second).ClusterSignalOverNoise->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ModuleLocalOccupancy "<<(idet->second).ModuleLocalOccupancy->getEntries()<<" "<<(idet->second).ModuleLocalOccupancy->getMean()<<" "<<(idet->second).ModuleLocalOccupancy->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ NrOfClusterizedStrips "<<(idet->second).NrOfClusterizedStrips->getEntries()<<" "<<(idet->second).NrOfClusterizedStrips->getMean()<<" "<<(idet->second).NrOfClusterizedStrips->getRMS()<<std::endl;
    }
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    // save histos in a file
     dqmStore_->save(outputFileName);
   }
}
//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = ClusterMEs.find(idet);
  ModMEs mod_me = pos->second;

  mod_me.NumberOfClusters->Reset();
  mod_me.ClusterPosition->Reset();
  mod_me.ClusterWidth->Reset();
  mod_me.ClusterCharge->Reset();
  mod_me.ClusterNoise->Reset();
  mod_me.ClusterSignalOverNoise->Reset();
  mod_me.ModuleLocalOccupancy->Reset();
  mod_me.NrOfClusterizedStrips->Reset(); // can be used at client level for occupancy calculations
}
