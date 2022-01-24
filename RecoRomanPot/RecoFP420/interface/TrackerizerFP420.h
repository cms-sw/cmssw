#ifndef TrackerizerFP420_h
#define TrackerizerFP420_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420TrackMain.h"

#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"

#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace cms {
  class TrackerizerFP420 : public edm::global::EDProducer<> {
  public:
    explicit TrackerizerFP420(const edm::ParameterSet& conf);
    //TrackerizerFP420();

    //  virtual void produce(ClusterCollectionFP420 &, TrackCollectionFP420 &);
    void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

  private:
    typedef std::vector<std::string> vstring;
    vstring trackerContainers;

    const FP420TrackMain sFP420TrackMain_;
    //  FP420TrackMain startFP420TrackMain_;
    //bool UseNoiseBadElectrodeFlagFromDB_;
    int verbosity;
  };
}  // namespace cms
#endif
