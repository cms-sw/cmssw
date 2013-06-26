#ifndef TkSeedGenerator_ClusterFilter_h
#define TkSeedGenerator_ClusterFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}

class ClusterFilter : public edm::EDFilter {
 public:
  ClusterFilter( const edm::ParameterSet & );
   ~ClusterFilter();
 private:
   bool filter( edm::Event &, edm::EventSetup const& );
   const int nMax_;
   const edm::ParameterSet  conf_;
   // int n_;
   // const bool verbose_;
};

#endif
