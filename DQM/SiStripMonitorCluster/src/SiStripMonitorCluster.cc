// -*- C++ -*-
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc
*/
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
// $Id: SiStripMonitorCluster.cc,v 1.28 2007/05/08 21:37:16 dkcira Exp $
#include <vector>
#include <numeric>
#include <fstream>
#include "TNamed.h"
#include "TH1F.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

//--------------------------------------------------------------------------------------------
SiStripMonitorCluster::SiStripMonitorCluster(const edm::ParameterSet& iConfig) : dbe_(edm::Service<DaqMonitorBEInterface>().operator->()), conf_(iConfig), SiStripNoiseService_(iConfig), show_mechanical_structure_view(true), show_readout_view(false), show_control_view(false), select_all_detectors(false), reset_each_run(false), fill_signal_noise (false) {} 
SiStripMonitorCluster::~SiStripMonitorCluster() { }

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::beginRun(const edm::Run&, const edm::EventSetup&){
  if(reset_each_run){ // reset histograms at beginning of each new run
    for(std::map<uint32_t, ModMEs>::const_iterator idet = ClusterMEs.begin(); idet!= ClusterMEs.end(); idet++ ){
     ResetME( (idet->second). NumberOfClusters );
     ResetME( (idet->second). ClusterPosition );
     ResetME( (idet->second). ClusterWidth );
     ResetME( (idet->second). ClusterCharge );
     ResetME( (idet->second). ClusterSignal );
     ResetME( (idet->second). ClusterNoise );
     ResetME( (idet->second). ClusterSignalOverNoise );
     ResetME( (idet->second). ModuleLocalOccupancy );
     ResetME( (idet->second). NrOfClusterizedStrips ); // can be used at client level for occupancy calculations
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
    charge_of_each_cluster = dbe_->book1D("ChargeOfEachCluster","ChargeOfEachCluster",500,-0.5,500.5);

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorCluster")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
    for(std::vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs modSingle;
      std::string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
      //nr. of clusters per module
      hid = hidmanager.createHistoId("NumberOfClusters","det",*detid_iterator);
      modSingle.NumberOfClusters = dbe_->book1D(hid, hid, 5,-0.5,4.5); dbe_->tag(modSingle.NumberOfClusters, *detid_iterator);
      modSingle.NumberOfClusters->setAxisTitle("number of clusters in one detector module");
      //ClusterPosition
      hid = hidmanager.createHistoId("ClusterPosition","det",*detid_iterator);
      modSingle.ClusterPosition = dbe_->book1D(hid, hid, 24,0.,768.); dbe_->tag(modSingle.ClusterPosition, *detid_iterator); // 6 APVs -> 768 strips
      modSingle.ClusterPosition->setAxisTitle("cluster position [strip number +0.5]");
      //ClusterWidth
      hid = hidmanager.createHistoId("ClusterWidth","det",*detid_iterator);
      modSingle.ClusterWidth = dbe_->book1D(hid, hid, 11,-0.5,10.5); dbe_->tag(modSingle.ClusterWidth, *detid_iterator);
      modSingle.ClusterWidth->setAxisTitle("cluster width [nr strips]");
      //ClusterCharge
      hid = hidmanager.createHistoId("ClusterCharge","det",*detid_iterator);
      modSingle.ClusterCharge = dbe_->book1D(hid, hid, 31,-0.5,300.5); dbe_->tag(modSingle.ClusterCharge, *detid_iterator);
      modSingle.ClusterCharge->setAxisTitle("cluster charge [ADC]");
      //ClusterNoise
      hid = hidmanager.createHistoId("ClusterNoise","det",*detid_iterator);
      modSingle.ClusterNoise = dbe_->book1D(hid, hid, 80,0.,10.); dbe_->tag(modSingle.ClusterNoise, *detid_iterator);
      modSingle.ClusterNoise->setAxisTitle("cluster noise");
      //ClusterSignal
      hid = hidmanager.createHistoId("ClusterSignal","det",*detid_iterator);
      modSingle.ClusterSignal = dbe_->book1D(hid, hid, 100,0.,300.); dbe_->tag(modSingle.ClusterSignal, *detid_iterator);
      modSingle.ClusterSignal->setAxisTitle("cluster signal");
      //ClusterSignalOverNoise
      hid = hidmanager.createHistoId("ClusterSignalOverNoise","det",*detid_iterator);
      modSingle.ClusterSignalOverNoise = dbe_->book1D(hid, hid, 100,0.,50.); dbe_->tag(modSingle.ClusterSignalOverNoise, *detid_iterator);
      modSingle.ClusterSignalOverNoise->setAxisTitle("ratio of signal to noise for each cluster");
      //ModuleLocalOccupancy
      hid = hidmanager.createHistoId("ModuleLocalOccupancy","det",*detid_iterator);
      // occupancy goes from 0 to 1, probably not over some limit value (here 0.1)
      modSingle.ModuleLocalOccupancy = dbe_->book1D(hid, hid, 20,-0.005,0.05); dbe_->tag(modSingle.ModuleLocalOccupancy, *detid_iterator);
      modSingle.ModuleLocalOccupancy->setAxisTitle("module local occupancy [% of clusterized strips]");
      //NrOfClusterizedStrips
      hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",*detid_iterator);
      modSingle.NrOfClusterizedStrips = dbe_->book1D(hid, hid, 10,-0.5,9.5); dbe_->tag(modSingle.NrOfClusterizedStrips, *detid_iterator);
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
   SiStripNoiseService_.setESObjects(iSetup);

  // retrieve producer name of input StripClusterCollection
  std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
  // get collection of DetSetVector of clusters from Event
  edm::Handle< edm::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByLabel(clusterProducer, cluster_detsetvektor);
  // auxiliary histogram with charge of each cluster
  for (edm::DetSetVector<SiStripCluster>::const_iterator icdetset=cluster_detsetvektor->begin();icdetset!=cluster_detsetvektor->end();icdetset++) {
    for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = (icdetset->data).begin(); clusterIter!= (icdetset->data).end(); clusterIter++){
      const std::vector<uint16_t>& ampls = clusterIter->amplitudes();
      short local_charge = 0;
      for(std::vector<uint16_t>::const_iterator iampls = ampls.begin(); iampls<ampls.end(); iampls++){
        local_charge += *iampls;
      }
      charge_of_each_cluster->Fill(static_cast<float>(local_charge),1.);
    }
  }
  // loop over MEs. Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
  for (std::map<uint32_t, ModMEs>::const_iterator iterMEs = ClusterMEs.begin() ; iterMEs!=ClusterMEs.end() ; iterMEs++) {
    uint32_t detid = iterMEs->first;  ModMEs modSingle = iterMEs->second;
    // get from DetSetVector the DetSet of clusters belonging to one detid - first make sure there exists clusters with this id
    edm::DetSetVector<SiStripCluster>::const_iterator isearch = cluster_detsetvektor->find(detid); // search  clusters of detid
    if(isearch==cluster_detsetvektor->end()){
      if(modSingle.NumberOfClusters != NULL){
        (modSingle.NumberOfClusters)->Fill(0.,1.); // no clusters for this detector module, so fill histogram with 0
      }
      continue; // no clusters for this detid => jump to next step of loop
    }
    //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
    edm::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'

    if(modSingle.NumberOfClusters != NULL){ // nr. of clusters per module
      (modSingle.NumberOfClusters)->Fill(static_cast<float>(cluster_detset.data.size()),1.);
    }
    if(modSingle.ClusterPosition != NULL){ // position of cluster
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
            (modSingle.ClusterPosition)->Fill(clusterIter->barycenter(),1.);
      }
    }
    short total_clusterized_strips = 0;
    if(modSingle.ClusterWidth != NULL){ // width of cluster, calculate yourself, no method for getting it
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
        const std::vector<uint16_t>& ampls = clusterIter->amplitudes();
        short local_size = ampls.size(); // width defined as nr. of strips that belong to cluster
        total_clusterized_strips = total_clusterized_strips + local_size; // add nr of strips of this cluster to total nr. of clusterized strips
        (modSingle.ClusterWidth)->Fill(static_cast<float>(local_size),1.);
      }
    }

    //
    float clusterSignal = 0;
    float clusterNoise = 0.;
    float clusterNoise2 = 0;
    int nrnonzeroamplitudes = 0;
    if( fill_signal_noise && (modSingle.ClusterSignalOverNoise || modSingle.ClusterSignal)){
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
        const std::vector<uint16_t>& ampls = clusterIter->amplitudes();
        for(uint iamp=0; iamp<ampls.size(); iamp++){
          if(ampls[iamp]>0){ // nonzero amplitude
            clusterSignal += ampls[iamp];
            try{
              if(!SiStripNoiseService_.getDisable(detid,clusterIter->firstStrip()+iamp)){
                  clusterNoise = SiStripNoiseService_.getNoise(detid,clusterIter->firstStrip()+iamp);
              }
            }catch(cms::Exception& e){
              edm::LogError("SiStripTkDQM|SiStripMonitorCluster|DB")<<" cms::Exception:  detid="<<detid<<" firstStrip="<<clusterIter->firstStrip()<<" iamp="<<iamp<<e.what();
            }
            clusterNoise2 += clusterNoise*clusterNoise;
            nrnonzeroamplitudes++;
          }
        }
        if(modSingle.ClusterSignal) (modSingle.ClusterSignal)->Fill(clusterSignal,1.);
        if(modSingle.ClusterNoise) (modSingle.ClusterNoise)->Fill(clusterNoise,1.);
        if(modSingle.ClusterSignalOverNoise) (modSingle.ClusterSignalOverNoise)->Fill(clusterSignal/sqrt(clusterNoise2/nrnonzeroamplitudes),1.);
      }
    }
    //
    if(modSingle.ClusterCharge != NULL){ // charge of cluster
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
        const std::vector<uint16_t>& ampls = clusterIter->amplitudes();
        short local_charge = 0;
        for(std::vector<uint16_t>::const_iterator iampls = ampls.begin(); iampls<ampls.end(); iampls++){
          local_charge += *iampls;
        }
        (modSingle.ClusterCharge)->Fill(static_cast<float>(local_charge),1.);
      }
    }
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
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterSignal "<<(idet->second).ClusterSignal->getEntries()<<" "<<(idet->second).ClusterSignal->getMean()<<" "<<(idet->second).ClusterSignal->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterNoise "<<(idet->second).ClusterNoise->getEntries()<<" "<<(idet->second).ClusterNoise->getMean()<<" "<<(idet->second).ClusterNoise->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ClusterSignalOverNoise "<<(idet->second).ClusterSignalOverNoise->getEntries()<<" "<<(idet->second).ClusterSignalOverNoise->getMean()<<" "<<(idet->second).ClusterSignalOverNoise->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ ModuleLocalOccupancy "<<(idet->second).ModuleLocalOccupancy->getEntries()<<" "<<(idet->second).ModuleLocalOccupancy->getMean()<<" "<<(idet->second).ModuleLocalOccupancy->getRMS()<<std::endl;
     monitor_summary<<"SiStripTkDQM|SiStripMonitorCluster"<<"              +++ NrOfClusterizedStrips "<<(idet->second).NrOfClusterizedStrips->getEntries()<<" "<<(idet->second).NrOfClusterizedStrips->getMean()<<" "<<(idet->second).NrOfClusterizedStrips->getRMS()<<std::endl;
    }
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    // save histos in a file
     dbe_->save(outputFileName);
   }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorCluster::ResetME(MonitorElement* me){
  MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
  if (ob) {
    TH1F * root_ob = dynamic_cast<TH1F *> (ob->operator->());
    if(root_ob)root_ob->Reset();
  } 
}
