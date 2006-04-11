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
// $Id: SiStripMonitorCluster.cc,v 1.5 2006/03/30 17:50:52 dkcira Exp $
//
//

#include <vector>
//#include <algorithm>
#include <numeric>

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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_mechanical_structure_view = "<<show_mechanical_structure_view;
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_readout_view = "<<show_readout_view;
   LogInfo("SiStripTkDQM|ConfigParams")<<"show_control_view = "<<show_control_view;

  if ( show_mechanical_structure_view ){
    // take from eventSetup the SiStripStructure object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripStructure> tkmechstruct;
    es.get<SiStripStructureRcd>().get(tkmechstruct);

    // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl object
    const vector<uint32_t> & activeDets = tkmechstruct->getActiveDetectorsRawIds();

    // use SiStripSubStructure for selecting certain regions
    SiStripSubStructure substructure;
    vector<uint32_t> SelectedDetIds;
    substructure.getTIBDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
    substructure.getTOBDetectors(activeDets, SelectedDetIds, 1, 2, 0);    // this adds rawDetIds to SelectedDetIds
    substructure.getTIDDetectors(activeDets, SelectedDetIds, 1, 1, 0, 0); // this adds rawDetIds to SelectedDetIds
    substructure.getTECDetectors(activeDets, SelectedDetIds, 1, 2, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds

    // use SistripHistoId for producing histogram id (and title)
    SiStripHistoId hidmanager;
    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;

    // loop over TOB detectors and book MEs
    LogInfo("SiStripTkDQM")<<"nr. of SelectedDetIds:  "<<SelectedDetIds.size();
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
      local_modmes.ClusterCharge = dbe_->book1D(hid, hid, 31,-0.5,256.5);
      //ModuleLocalOccupancy
      hid = hidmanager.createHistoId("ModuleLocalOccupancy","det",*detid_iterator);
      local_modmes.ModuleLocalOccupancy = dbe_->book1D(hid, hid, 20,0.,1.0);// occupancy goes from 0 to 1, probably not over some limit value (here 0.5)
      //NrOfClusterizedStrips
      hid = hidmanager.createHistoId("NrOfClusterizedStrips","det",*detid_iterator);
      local_modmes.NrOfClusterizedStrips = dbe_->book1D(hid, hid, 20,0.,768.);
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
    const SiStripClusterCollection::Range clusterRange = cluster_collection->get(detid);
//    SiStripClusterCollection::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
//    SiStripClusterCollection::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
    SiStripClusterCollection::ContainerIterator icluster;


    if(local_modmes.NrClusters != NULL){ // nr. of clusters per module
      // following line works only if clusters consecutive but is much shorter than looping
      int nr_clusters = clusterRange.second - clusterRange.first + 1;
      (local_modmes.NrClusters)->Fill(static_cast<float>(nr_clusters),1.);
    }
    if(local_modmes.ClusterPosition != NULL){ // position of cluster
      for(icluster = clusterRange.first; icluster<clusterRange.second; icluster++){
        (local_modmes.ClusterPosition)->Fill((*icluster).barycenter(),1.);
      }
    }
    short total_clusterized_strips = 0;
    if(local_modmes.ClusterWidth != NULL){ // width of cluster
//--- ! no method for getting directly width
      for(icluster = clusterRange.first; icluster<clusterRange.second; icluster++){
        const std::vector<short>& ampls = icluster->amplitudes();
        short local_size = ampls.size(); // nr. of strips that belong to cluster - use this as width for the moment
        total_clusterized_strips = total_clusterized_strips + local_size; // add nr of strips of this cluster to total nr. of clusterized strips
        (local_modmes.ClusterWidth)->Fill(static_cast<float>(local_size),1.);
      }
    }
    if(local_modmes.ClusterCharge != NULL){ // charge of cluster
      for(icluster = clusterRange.first; icluster<clusterRange.second; icluster++){
        const std::vector<short>& ampls = icluster->amplitudes();
//        short local_charge = accumulate( ampls.begin(), ampls.end(), 0 ); // when using this program crashes
        short local_charge = 0;
        for(std::vector<short>::const_iterator i = ampls.begin(); i<ampls.end(); i++){
          local_charge += *i;
        }
        (local_modmes.ClusterCharge)->Fill(static_cast<float>(local_charge),1.);
      }
    }
    local_modmes.NrOfClusterizedStrips->Fill(static_cast<float>(total_clusterized_strips),1.);
    short total_nr_strips = 6 * 128; // assume 6 APVs per detector for the moment. later ask FedCabling object
    float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
    local_modmes.ModuleLocalOccupancy->Fill(local_occupancy,1.);
//
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
