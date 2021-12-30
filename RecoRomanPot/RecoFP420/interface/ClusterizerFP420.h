#ifndef ClusterizerFP420_h
#define ClusterizerFP420_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
//#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420ClusterMain.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"

#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace cms {
  class ClusterizerFP420 : public edm::global::EDProducer<> {
  public:
    explicit ClusterizerFP420(const edm::ParameterSet& conf);

    void beginJob() override;

    void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

  private:
    typedef std::vector<std::string> vstring;

    vstring trackerContainers;

    std::unique_ptr<const FP420ClusterMain> sClusterizerFP420_;

    bool UseNoiseBadElectrodeFlagFromDB_;
    int sn0, pn0, dn0, rn0;
    int verbosity;
  };
}  // namespace cms
#endif
