#ifndef Fireworks_Tracks_FWBeamSpot_h
#define Fireworks_Tracks_FWBeamSpot_h

#include "DataFormats/BeamSpot/interface/BeamSpotFwd.h"

namespace edm {
  class EventBase;
}

class FWBeamSpot {
public:
  FWBeamSpot() : m_beamspot(nullptr) {}
  ~FWBeamSpot() {}

  void checkBeamSpot(const edm::EventBase* event);

  double x0() const;
  double y0() const;
  double z0() const;
  double x0Error() const;
  double y0Error() const;
  double z0Error() const;

  const reco::BeamSpot* getBeamSpot() const { return m_beamspot; }

private:
  const reco::BeamSpot* m_beamspot;
};

#endif
