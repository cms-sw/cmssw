#ifndef MuonIsolation_MuIsoBaseAlgorithm_H
#define MuonIsolation_MuIsoBaseAlgorithm_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace edm { class Event; }
namespace edm { class EventSetup; }

class MuIsoBaseAlgorithm {
public:
  /// Destructor
  virtual ~MuIsoBaseAlgorithm() {}

  /// The isolation result for one muon
  virtual float isolation(const edm::Event&, 
			  const edm::EventSetup&, 
			  const reco::Track& muon) = 0;
  virtual float isolation(const edm::Event&, const edm::EventSetup&, const reco::TrackRef& muon) = 0;

  /// Return logical result of isolaton is all parameters and cuts are fixe
  /// (may remain not implemented for all types of isolation)
  virtual bool isIsolated(const edm::Event&, const edm::EventSetup&, const reco::Track& muon) = 0;
  virtual bool isIsolated(const edm::Event&, const edm::EventSetup&, const reco::TrackRef& muon) = 0;

  /// The component that reconstructs and selects deposits
  //virtual MuIsoExtractor * extractor() = 0;

  /// The component that computes the isolation value from the deposits
  //virtual MuIsoBaseIsolator  * isolator() = 0;

};
#endif
