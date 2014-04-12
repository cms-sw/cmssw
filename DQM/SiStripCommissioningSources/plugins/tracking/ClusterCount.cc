#include "DQM/SiStripCommissioningSources/plugins/tracking/ClusterCount.h"

//
// constructors and destructor
//
ClusterCount::ClusterCount(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  //   clusterLabel_ = iConfig.getParameter<edm::InputTag>("ClustersLabel");
  clusterToken_ = consumes<edm::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("ClustersLabel") );
   
}


ClusterCount::~ClusterCount()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ClusterCount::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   // look at the clusters
   edm::Handle<edm::DetSetVector<SiStripCluster> > clusters;
   iEvent.getByToken(clusterToken_, clusters);
   const edm::DetSetVector<SiStripCluster>* clusterSet = clusters.product();
   // loop on the detsetvector<cluster>
   for (edm::DetSetVector<SiStripCluster>::const_iterator DSViter=clusterSet->begin(); DSViter!=clusterSet->end();DSViter++ ) {
     edm::DetSet<SiStripCluster>::const_iterator begin=DSViter->data.begin();
     edm::DetSet<SiStripCluster>::const_iterator end  =DSViter->data.end();
     for(edm::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
       LogDebug("ReconstructedClusters") << "Detid/Strip: " << std::hex << DSViter->id << std::dec << " / " << iter->barycenter();
     }
   }
}

