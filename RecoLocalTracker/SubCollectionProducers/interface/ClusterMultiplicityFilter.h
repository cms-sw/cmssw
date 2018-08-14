#ifndef ClusterMultiplicityFilter_h
#define ClusterMultiplicityFilter_h

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


class ClusterMultiplicityFilter : public edm::global::EDFilter<> {
   public:
      explicit ClusterMultiplicityFilter(const edm::ParameterSet&);
      ~ClusterMultiplicityFilter() override;

   private:
      bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

      const unsigned int maxNumberOfClusters_;
      const edm::InputTag clusterCollectionTag_;
      const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_;

};

#endif
