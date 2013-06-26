#ifndef ClusterMultiplicityFilter_h
#define ClusterMultiplicityFilter_h

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class ClusterMultiplicityFilter : public edm::EDFilter {
   public:
      explicit ClusterMultiplicityFilter(const edm::ParameterSet&);
      ~ClusterMultiplicityFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      unsigned int maxNumberOfClusters_;
      std::string clusterCollectionLabel_; 

};

#endif
