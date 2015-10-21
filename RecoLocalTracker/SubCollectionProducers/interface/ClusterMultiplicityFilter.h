#ifndef ClusterMultiplicityFilter_h
#define ClusterMultiplicityFilter_h

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


class ClusterMultiplicityFilter : public edm::stream::EDFilter<> {
   public:
      explicit ClusterMultiplicityFilter(const edm::ParameterSet&);
      ~ClusterMultiplicityFilter();

   private:

      virtual bool filter(edm::Event&, const edm::EventSetup&);

      const unsigned int maxNumberOfClusters_;
      const std::string clusterCollectionLabel_;

      edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_;

};

#endif
