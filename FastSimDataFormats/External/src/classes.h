#include "FastSimDataFormats/External/interface/FastL1BitInfo.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary { // dummy is a dummy
    
    FastL1BitInfoCollection dummy01;
    edm::Wrapper<FastL1BitInfoCollection> dummy02;
    
    FastTrackerCluster d1;
    FastTrackerClusterCollection d2;
    edm::Wrapper<FastTrackerClusterCollection> d3;

    edm::ClonePolicy<FastTrackerCluster>  d4;  
    edm::OwnVector<FastTrackerCluster,edm::ClonePolicy<FastTrackerCluster> > d5;   

    edm::Wrapper< edm::RangeMap<unsigned int,
      edm::OwnVector<FastTrackerCluster,
      edm::ClonePolicy<FastTrackerCluster> >, 
      edm::ClonePolicy<FastTrackerCluster> > >  d7;
      
    edm::Ref<edm::RangeMap<unsigned int,edm::OwnVector<FastTrackerCluster,
      edm::ClonePolicy<FastTrackerCluster> >,edm::ClonePolicy<FastTrackerCluster> >,FastTrackerCluster,
      edm::refhelper::FindUsingAdvance<edm::RangeMap<unsigned int,edm::OwnVector<FastTrackerCluster,
      edm::ClonePolicy<FastTrackerCluster> >,edm::ClonePolicy<FastTrackerCluster> >,FastTrackerCluster> > d8;

  }; 
}
