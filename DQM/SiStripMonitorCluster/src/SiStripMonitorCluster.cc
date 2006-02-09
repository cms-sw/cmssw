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
// $Id$
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h" // these two will go away
#include "CalibTracker/Records/interface/SiStripStructureRcd.h"     // these two will go away

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoIdManager.h"
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

    // select (certain) TOB detectors from activeDets and put them in TOBDetIds
    vector<uint32_t> TOBDetIds;
    SiStripSubStructure substructure;
    // select TOBs of layer=1. 0 selects everything
    substructure.getTOBDetectors(activeDets, TOBDetIds, 1, 3, 4, 0);

    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;
    // loop over TOB detectors and book MEs
    cout<<"SiStripMonitorCluster::analyze nr. of TOBDetIds:  "<<TOBDetIds.size()<<endl;
    for(vector<uint32_t>::const_iterator detid_iterator = TOBDetIds.begin(); detid_iterator!=TOBDetIds.end(); detid_iterator++){
      // use SistripHistoIdManager for producing histogram id (and title)
      SiStripHistoIdManager hidmanager;
      string hid = hidmanager.createHistoId("clusters", *detid_iterator);
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(*detid_iterator); // pass the detid to this method
      // book ME
      MonitorElement* local_me = dbe_->book1D(hid, hid, 31,-0.5,30.5);
      NrClusters.insert( pair<uint32_t, MonitorElement*>(*detid_iterator,local_me) );
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
  for (map<uint32_t, MonitorElement*>::const_iterator i = NrClusters.begin() ; i!=NrClusters.end() ; i++) {
    uint32_t detid = i->first;
    MonitorElement* local_me = i->second;
    // retrieve producer name of input StripClusterCollection
    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    // get ClusterCollection object from Event
    edm::Handle<SiStripClusterCollection> cluster_collection;
    iEvent.getByLabel(clusterProducer, cluster_collection);
    // get range of clusters belonging to detector detid
    const SiStripClusterCollection::Range cluster_range = cluster_collection->get(detid);
    // following line works only if clusters consecutive but is much shorter than looping
    int nr_clusters = cluster_range.second - cluster_range.first + 1;
    local_me->Fill( float(nr_clusters), 1. );
  }
}


void SiStripMonitorCluster::endJob(void){
//  dbe_->showDirStructure();
  dbe_->save("test_cluster.root");
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorCluster)
