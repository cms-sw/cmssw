#ifndef L2TauPixelTrackMatch_h
#define L2TauPixelTrackMatch_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <vector>

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

/** class L2TauPixelTrackMatch
 * this producer creates a new L2 tau jet collection with jets' vertices redefined 
 * from vertex z (relative to beamspot) of dr-matched pixel tracks 
 * that are above some pt threshold and beamline x & y.
 */
class L2TauPixelTrackMatch : public edm::global::EDProducer<> {
public:
  explicit L2TauPixelTrackMatch(const edm::ParameterSet&);
  ~L2TauPixelTrackMatch() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  struct TinyTrack {
    float pt, eta, phi;
    math::XYZPoint vtx;
  };

  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_jetSrc;
  float m_jetMinPt;
  float m_jetMaxEta;
  edm::EDGetTokenT<reco::TrackCollection> m_trackSrc;
  float m_trackMinPt;
  float m_deltaR;
  edm::EDGetTokenT<reco::BeamSpot> m_beamSpotTag;
};

#endif
