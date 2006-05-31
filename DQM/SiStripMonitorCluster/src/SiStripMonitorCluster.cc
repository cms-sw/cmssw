// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorCluster
// 
/**\class SiStripMonitorCluster SiStripMonitorCluster.cc DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dorian Kcira
//         Created:  Wed Feb  1 16:42:34 CET 2006
// $Id: SiStripMonitorCluster.cc,v 1.10 2006/05/23 13:35:00 dkcira Exp $
//
//

#include <vector>
//#include <algorithm>
#include <numeric>
#include <iostream>

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

using namespace std;
using namespace edm;

SiStripMonitorCluster::SiStripMonitorCluster(const edm::ParameterSet& iConfig)
{
   dbe_  = edm::Service<DaqMonitorBEInterface>().operator->();
   conf_ = iConfig;
}


SiStripMonitorCluster::~SiStripMonitorCluster()
{
}


void SiStripMonitorCluster::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file
   bool show_mechanical_structure_view = conf_.getParameter<bool>("ShowMechanicalStructureView");
   bool show_readout_view = conf_.getParameter<bool>("ShowReadoutView");
   bool show_control_view = conf_.getParameter<bool>("ShowControlView");
   bool select_all_detectors = conf_.getParameter<bool>("SelectAllDetectors");
   LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowMechanicalStructureView = "<<show_mechanical_structure_view;
   LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowReadoutView = "<<show_readout_view;
   LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"ShowControlView = "<<show_control_view;
   LogInfo("SiStripTkDQM|SiStripMonitorCluster|ConfigParams")<<"SelectAllDetectors = "<<select_all_detectors;

  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    es.get<SiStripDetCablingRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
    vector<uint32_t> activeDets;
    activeDets.clear(); // just in case
    tkmechstruct->addActiveDetectorsRawIds(activeDets);

    vector<uint32_t> SelectedDetIds;
    if(select_all_detectors){
      // select all detectors if appropriate flag is set,  for example for the mtcc
      SelectedDetIds = activeDets;
    }else{
      // use SiStripSubStructure for selecting certain regions
      SiStripSubStructure substructure;
      substructure.getTIBDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
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

    // loop over TOB detectors and book MEs
    LogInfo("SiStripTkDQM|SiStripMonitorCluster")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
    for(vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs local_modmes;
      string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
      //nr. of clusters per module
      hid = hidmanager.createHistoId("ClustersPerDetector","det",*detid_iterator);
      local_modmes.NrClusters = dbe_->book1D(hid, hid, 31,-0.5,30.5);
      //ClusterPosition
      hid = hidmanager.createHistoId("ClusterPosition","det",*detid_iterator);
      local_modmes.ClusterPosition = dbe_->book1D(hid, hid, 30,-0.5,128.5);
      //ClusterWidth
      hid = hidmanager.createHistoId("ClusterWidth","det",*detid_iterator);
      local_modmes.ClusterWidth = dbe_->book1D(hid, hid, 10,-0.5,10.5);
      //ClusterWidth
      hid = hidmanager.createHistoId("ClusterCharge","det",*detid_iterator);
      local_modmes.ClusterCharge = dbe_->book1D(hid, hid, 31,-0.5,30.5);
      //ModuleLocalOccupancy
      hid = hidmanager.createHistoId("ModuleLocalOccupancy","det",*detid_iterator);
      local_modmes.ModuleLocalOccupancy = dbe_->book1D(hid, hid, 20,0.,0.1);// occupancy goes from 0 to 1, probably not over some limit value (here 0.1)
      //NrOfClusterizedStrips
      hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",*detid_iterator);
      local_modmes.NrOfClusterizedStrips = dbe_->book1D(hid, hid, 21,-0.,19.5);
      // append to ClusterMEs
      ClusterMEs.insert( std::make_pair(*detid_iterator, local_modmes));
    }
  }
// below is just for testing
//      SiStripHistoIdManager hidmanager2;
//      uint32_t cid3 = hidmanager2.getComponentId("mbrame _#_3433");
//      uint32_t cid4 = hidmanager2.getComponentId("stre;lkjasdmbrame _#_21234444");
//      uint32_t cid2 = hidmanager2.getComponentId("la la _ _ # stra fu");
}


void SiStripMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  // retrieve producer name of input StripClusterCollection
  std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
  // get collection of DetSetVector of clusters from Event
  edm::Handle< edm::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByLabel(clusterProducer, cluster_detsetvektor);
//  std::cout<<"cluster_detsetvektor.size()="<<cluster_detsetvektor->size()<<endl;
  // loop over MEs. Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
  for (map<uint32_t, ModMEs>::const_iterator iterMEs = ClusterMEs.begin() ; iterMEs!=ClusterMEs.end() ; iterMEs++) {
    uint32_t detid = iterMEs->first;  ModMEs local_modmes = iterMEs->second;
    // get from DetSetVector the DetSet of clusters belonging to one detid - first make sure there exists clusters with this id
    edm::DetSetVector<SiStripCluster>::const_iterator isearch = cluster_detsetvektor->find(detid); // search  clusters of detid
    if(isearch==cluster_detsetvektor->end()) continue; // no clusters for this detid => jump to next step of loop
    //cluster_detset is a structure, cluster_detset.data is a std::vector<SiStripCluster>, cluster_detset.id is uint32_t
    edm::DetSet<SiStripCluster> cluster_detset = (*cluster_detsetvektor)[detid]; // the statement above makes sure there exists an element with 'detid'

    if(local_modmes.NrClusters != NULL){ // nr. of clusters per module
//      std::cout<<"detid="<<detid<<" cluster_detset.data.size()="<<cluster_detset.data.size()<<endl;
      (local_modmes.NrClusters)->Fill(static_cast<float>(cluster_detset.data.size()),1.);
    }
    if(local_modmes.ClusterPosition != NULL){ // position of cluster
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
//            std::cout<<"                  cluster_Iter->barycenter()"<<clusterIter->barycenter()<<endl;
            (local_modmes.ClusterPosition)->Fill(clusterIter->barycenter(),1.);
      }
    }
    short total_clusterized_strips = 0;
    if(local_modmes.ClusterWidth != NULL){ // width of cluster, calculate yourself, no method for getting it
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
        const std::vector<short>& ampls = clusterIter->amplitudes();
        short local_size = ampls.size(); // width defined as nr. of strips that belong to cluster
        total_clusterized_strips = total_clusterized_strips + local_size; // add nr of strips of this cluster to total nr. of clusterized strips
        (local_modmes.ClusterWidth)->Fill(static_cast<float>(local_size),1.);
//        std::cout<<"                  cluster_width"<<local_size<<endl;
      }
    }
    if(local_modmes.ClusterCharge != NULL){ // charge of cluster
      for(edm::DetSet<SiStripCluster>::const_iterator clusterIter = cluster_detset.data.begin(); clusterIter!= cluster_detset.data.end(); clusterIter++){
        const std::vector<short>& ampls = clusterIter->amplitudes();
        short local_charge = 0;
        for(std::vector<short>::const_iterator iampls = ampls.begin(); iampls<ampls.end(); iampls++){
          local_charge += *iampls;
        }
        (local_modmes.ClusterCharge)->Fill(static_cast<float>(local_charge),1.);
//        std::cout<<"                  cluster_charge"<<local_charge<<endl;
      }
    }
    if(local_modmes.NrOfClusterizedStrips != NULL){ // nr of clusterized strips
      local_modmes.NrOfClusterizedStrips->Fill(static_cast<float>(total_clusterized_strips),1.);
    }
    short total_nr_strips = 6 * 128; // assume 6 APVs per detector for the moment. later ask FedCabling object
    float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
    if(local_modmes.ModuleLocalOccupancy != NULL){ // nr of clusterized strips
      local_modmes.ModuleLocalOccupancy->Fill(local_occupancy,1.);
    }
  }
}


void SiStripMonitorCluster::endJob(void){
    bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
    string outputFileName = conf_.getParameter<string>("OutputFileName");
 //  dbe_->showDirStructure();
   if(outputMEsInRootFile){
     dbe_->save(outputFileName);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorCluster)
