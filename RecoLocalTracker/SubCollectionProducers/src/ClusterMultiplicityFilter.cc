// -*- C++ -*-
//
// Package:    ClusterMultiplicityFilter
// Class:      ClusterMultiplicityFilter
// 

//
// Original Author:  Carsten Noeding
//         Created:  Mon Mar 19 13:51:22 CDT 2007
//
//


// system include files
#include <memory>

#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterMultiplicityFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


ClusterMultiplicityFilter::ClusterMultiplicityFilter(const edm::ParameterSet& iConfig) :
  maxNumberOfClusters_(iConfig.getParameter<unsigned int>("MaxNumberOfClusters")),
  clusterCollectionTag_(iConfig.getParameter<edm::InputTag>("ClusterCollection")),
  clusters_ (consumes<edmNew::DetSetVector<SiStripCluster> >(clusterCollectionTag_))
{

}


ClusterMultiplicityFilter::~ClusterMultiplicityFilter() {
}


// ------------ method called on each new Event  ------------
bool ClusterMultiplicityFilter::filter(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {

  bool result = true;

  const edmNew::DetSetVector<SiStripCluster> *clusters = nullptr;
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
  iEvent.getByToken(clusters_,clusterHandle);
  if( !clusterHandle.isValid() ) {
    throw cms::Exception("CorruptData")
      << "ClusterMultiplicityFilter requires collection <edm::DetSetVector<SiStripCluster> with label " << clusterCollectionTag_.label() << std::endl;
  }

  clusters = clusterHandle.product();
  const edmNew::DetSetVector<SiStripCluster>& input = *clusters;

  unsigned int totalClusters = 0;

  //loop over detectors
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(); DSViter!=input.end();DSViter++ ) {
    totalClusters+=DSViter->size();
  }
  
  
  if (totalClusters>maxNumberOfClusters_) {
    edm::LogInfo("ClusterMultiplicityFilter") << "Total number of clusters: " << totalClusters << " ==> exceeds allowed maximum of " << maxNumberOfClusters_ << " clusters";
    result = false;
  }
  
  return result;
}
