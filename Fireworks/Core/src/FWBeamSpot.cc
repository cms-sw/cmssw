#include "Fireworks/Core/interface/FWBeamSpot.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Common/interface/EventBase.h"
#include "Fireworks/Core/interface/fwLog.h"

void FWBeamSpot::checkBeamSpot(const edm::EventBase* event) {
  try {
    edm::InputTag tag("offlineBeamSpot");
    edm::Handle<reco::BeamSpot> spot;

    event->getByLabel(tag, spot);
    if (spot.isValid()) {
      m_beamspot = spot.product();
    } else {
      m_beamspot = nullptr;
    }
  } catch (cms::Exception& iException) {
    fwLog(fwlog::kWarning) << "Can't get beam spot info. Setting coordintes to (0, 0, 0).\n";
    m_beamspot = nullptr;
  }
}

double FWBeamSpot::x0() const { return m_beamspot ? m_beamspot->x0() : 0.0; }

double FWBeamSpot::y0() const { return m_beamspot ? m_beamspot->y0() : 0.0; }

double FWBeamSpot::z0() const { return m_beamspot ? m_beamspot->z0() : 0.0; }

double FWBeamSpot::x0Error() const { return m_beamspot ? m_beamspot->x0Error() : 0.0; }

double FWBeamSpot::y0Error() const { return m_beamspot ? m_beamspot->y0Error() : 0.0; }

double FWBeamSpot::z0Error() const { return m_beamspot ? m_beamspot->z0Error() : 0.0; }
