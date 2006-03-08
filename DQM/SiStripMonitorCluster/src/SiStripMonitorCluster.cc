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
// $Id: SiStripMonitorCluster.cc,v 1.1 2006/02/09 19:28:56 gbruno Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h" // these two will go away
#include "CalibTracker/Records/interface/SiStripStructureRcd.h"     // these two will go away

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

using namespace std;

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
   cout<<"SiStripMonitorCluster::beginJob : show_mechanical_structure_view = "<<show_mechanical_structure_view<<endl;
   cout<<"SiStripMonitorCluster::beginJob : show_readout_view = "<<show_readout_view<<endl;
   cout<<"SiStripMonitorCluster::beginJob : show_control_view = "<<show_control_view<<endl;

  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripStructure object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripStructure> tkmechstruct;
    es.get<SiStripStructureRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl object
    const vector<uint32_t> & activeDets = tkmechstruct->getActiveDetectorsRawIds();

    // use SiStripSubStructure for selecting certain regions
    SiStripSubStructure substructure;
    vector<uint32_t> SelectedDetIds;
    // select TIBs of layer=2. 0 selects everything
    substructure.getTIBDetectors(activeDets, SelectedDetIds, 2, 7, 2, 0); // this adds rawDetIds to SelectedDetIds
    // select TOBs of layer=1, etc.
    substructure.getTOBDetectors(activeDets, SelectedDetIds, 1, 3, 4, 0); // this adds rawDetIds to SelectedDetIds

    // use SistripHistoId for producing histogram id (and title)
    SiStripHistoId hidmanager;
    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;

    // loop over TOB detectors and book MEs
    cout<<"SiStripMonitorCluster::analyze nr. of SelectedDetIds:  "<<SelectedDetIds.size()<<endl;
    for(vector<uint32_t>::const_iterator detid_iterator = SelectedDetIds.begin(); detid_iterator!=SelectedDetIds.end(); detid_iterator++){
      ModMEs local_modmes;
      string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
      // create nr. of clusters per module
      hid = hidmanager.createHistoId("ClusterDistribution","det",*detid_iterator);
      local_modmes.NrClusters = dbe_->book1D(hid, hid, 31,-0.5,30.5);
      //create ClusterPosition
      hid = hidmanager.createHistoId("ClusterPosition","det",*detid_iterator);
      local_modmes.ClusterPosition = dbe_->book1D(hid, hid, 31,-0.5,30.5);
      //create ClusterWidth
      hid = hidmanager.createHistoId("ClusterWidth","det",*detid_iterator);
      local_modmes.ClusterWidth = dbe_->book1D(hid, hid, 31,-0.5,30.5);
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
  // Mechanical structure view. No need for condition here. If map is empty, nothing should happen.
  for (map<uint32_t, ModMEs>::const_iterator i = ClusterMEs.begin() ; i!=ClusterMEs.end() ; i++) {
    uint32_t detid = i->first;  ModMEs local_modmes = i->second;

    // retrieve producer name of input StripClusterCollection
    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    // get ClusterCollection object from Event
    edm::Handle<SiStripClusterCollection> cluster_collection;
    iEvent.getByLabel(clusterProducer, cluster_collection);
    // get range of clusters belonging to detector detid
    const SiStripClusterCollection::Range cluster_range = cluster_collection->get(detid);

    if(local_modmes.NrClusters != NULL){ // nr. of clusters per module
      // following line works only if clusters consecutive but is much shorter than looping
      int nr_clusters = cluster_range.second - cluster_range.first + 1;
      (local_modmes.NrClusters)->Fill(static_cast<float>(nr_clusters),1.);
    }
    if(local_modmes.ClusterPosition != NULL){ // position of cluster
//      for(SiStripClusterCollection::iterator icluster = cluster_range.first; icluster<cluster_range.second; icluster++){
//      (local_modmes.ClusterPosition)->Fill((*icluster).barycenter(),1.);
//      }
    }
    if(local_modmes.ClusterWidth != NULL){ // width of cluster
//--- ! no method for getting directly width - leave empty for the moment
//      for(SiStripClusterCollection::const_iterator icluster = cluster_range.first; icluster<cluster_range.second; icluster++){
//      (local_modmes.ClusterWidth)->Fill((*icluster).barycenter(),1.);
//      }
    }
  }
}

void SiStripMonitorCluster::endJob(void){
//  dbe_->showDirStructure();
  dbe_->save("test_cluster.root");
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorCluster)
