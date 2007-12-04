// -*- C++ -*-
//
// Package:    ClusterMultiplicityFilter
// Class:      ClusterMultiplicityFilter
// 

//
// Original Author:  Carsten Noeding
//         Created:  Mon Mar 19 13:51:22 CDT 2007
// $Id: ClusterMultiplicityFilter.cc,v 1.1 2007/03/31 08:46:35 noeding Exp $
//
//


// system include files
#include <memory>

#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterMultiplicityFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


ClusterMultiplicityFilter::ClusterMultiplicityFilter(const edm::ParameterSet& iConfig)
{
  maxNumberOfClusters_    = iConfig.getUntrackedParameter<unsigned int>("MaxNumberOfClusters");
  clusterCollectionLabel_ = iConfig.getUntrackedParameter<std::string>("ClusterCollectionLabel");
}


ClusterMultiplicityFilter::~ClusterMultiplicityFilter() {
}


// ------------ method called on each new Event  ------------
bool ClusterMultiplicityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  bool result = true;

  const edm::DetSetVector<SiStripCluster> *clusters = 0;
  try {
    edm::Handle<edm::DetSetVector<SiStripCluster> > clusterHandle;
    iEvent.getByLabel(clusterCollectionLabel_,clusterHandle);
    clusters = clusterHandle.product();
  }
  catch (edm::Exception const& x) {
    if ( x.categoryCode() == edm::errors::ProductNotFound ) {
      if ( x.history().size() == 1 ) {
	edm::LogWarning("ClusterMultiplicityFilter") << "Collection <edm::DetSetVector<SiStripCluster> with label " << clusterCollectionLabel_ << " cannot be found.";
      }
    }
  }

  const edm::DetSetVector<SiStripCluster>& input = *clusters;

  unsigned int totalClusters = 0;

  //loop over detectors
  for (edm::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(); DSViter!=input.end();DSViter++ ) {
    totalClusters+=DSViter->data.size();
  }
  
  
  if (totalClusters>maxNumberOfClusters_) {
    edm::LogInfo("ClusterMultiplicityFilter") << "Total number of clusters: " << totalClusters << " ==> exceeds allowed maximum of " << maxNumberOfClusters_ << " clusters";
    result = false;
  }
  
  return result;
}


void ClusterMultiplicityFilter::beginJob(const edm::EventSetup&) {
}


// ------------ method called once each job just after ending the event loop  ------------
void ClusterMultiplicityFilter::endJob() {
}

