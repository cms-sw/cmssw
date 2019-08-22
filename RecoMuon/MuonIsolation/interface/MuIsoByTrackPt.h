#ifndef MuonIsolation_MuIsoByTrackPt_H
#define MuonIsolation_MuIsoByTrackPt_H

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseAlgorithm.h"
#include "RecoMuon/MuonIsolation/interface/CutsConeSizeFunction.h"

namespace reco {
  namespace isodeposit {
    class IsoDepositExtractor;
  }
}  // namespace reco
namespace muonisolation {
  class IsolatorByDeposit;
}
namespace reco {
  class Track;
}
namespace edm {
  class Event;
}
namespace edm {
  class EventSetup;
}
namespace edm {
  class ConsumesCollector;
}
namespace edm {
  class ParameterSet;
}

class MuIsoByTrackPt : public MuIsoBaseAlgorithm {
public:
  MuIsoByTrackPt(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);
  ~MuIsoByTrackPt() override;

  float isolation(const edm::Event&, const edm::EventSetup&, const reco::Track& muon) override;
  float isolation(const edm::Event& ev, const edm::EventSetup& es, const reco::TrackRef& muon) override {
    return isolation(ev, es, *muon);
  }
  bool isIsolated(const edm::Event&, const edm::EventSetup&, const reco::Track& muon) override;
  bool isIsolated(const edm::Event& ev, const edm::EventSetup& es, const reco::TrackRef& muon) override {
    return isIsolated(ev, es, *muon);
  }

  void setConeSize(float dr);
  void setCut(float cut) { theCut = cut; }

  virtual reco::isodeposit::IsoDepositExtractor* extractor() { return theExtractor.get(); }
  virtual muonisolation::IsolatorByDeposit* isolator() { return theIsolator.get(); }

private:
  float theCut;
  std::unique_ptr<reco::isodeposit::IsoDepositExtractor> theExtractor;
  std::unique_ptr<muonisolation::IsolatorByDeposit> theIsolator;
};

#endif
