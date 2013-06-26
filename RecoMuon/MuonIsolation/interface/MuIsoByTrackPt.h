#ifndef MuonIsolation_MuIsoByTrackPt_H
#define MuonIsolation_MuIsoByTrackPt_H

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseAlgorithm.h"
#include "RecoMuon/MuonIsolation/interface/CutsConeSizeFunction.h"

namespace reco { namespace isodeposit { class IsoDepositExtractor; }}
namespace muonisolation { class IsolatorByDeposit; }
namespace reco { class Track; }
namespace edm { class Event; }
namespace edm { class EventSetup; }
namespace edm { class ParameterSet; }


class MuIsoByTrackPt : public MuIsoBaseAlgorithm {
public:
  MuIsoByTrackPt(const edm::ParameterSet& conf);
  virtual ~MuIsoByTrackPt();
   
  virtual float isolation(const edm::Event&, const edm::EventSetup&, const reco::Track& muon);
  virtual float isolation(const edm::Event& ev, const edm::EventSetup& es, const reco::TrackRef& muon)
  {
    return isolation(ev, es, *muon);
  }
  virtual bool isIsolated(const edm::Event&, const edm::EventSetup&, const reco::Track& muon);
  virtual bool isIsolated(const edm::Event& ev, const edm::EventSetup& es, const reco::TrackRef& muon)
  {
    return isIsolated(ev, es, *muon);
  }

  void setConeSize(float dr);
  void setCut(float cut) { theCut = cut; }
  
  virtual reco::isodeposit::IsoDepositExtractor * extractor() { return theExtractor; }
  virtual muonisolation::IsolatorByDeposit * isolator() { return theIsolator; }

private:
  float theCut;
  reco::isodeposit::IsoDepositExtractor * theExtractor;
  muonisolation::IsolatorByDeposit * theIsolator;
};

#endif
