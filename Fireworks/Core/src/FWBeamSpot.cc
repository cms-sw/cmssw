#include "Fireworks/Core/interface/FWBeamSpot.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Common/interface/EventBase.h"

void FWBeamSpot::checkBeamSpot(const edm::EventBase* event)
{
   edm::InputTag tag("offlineBeamSpot");
   edm::Handle<reco::BeamSpot> spot;

   event->getByLabel(tag, spot);
   if (spot.isValid())
   {
      m_beamspot = spot.product();
   }
   else
   {
      m_beamspot = 0;
   }
}

double FWBeamSpot::x0() const
{
   return m_beamspot ? m_beamspot->x0() : 0.0;
}

double FWBeamSpot::y0() const
{
   return m_beamspot ? m_beamspot->y0() : 0.0;
}

double FWBeamSpot::z0() const
{
   return m_beamspot ? m_beamspot->z0() : 0.0;
}

double FWBeamSpot::x0Error() const
{
   return m_beamspot ? m_beamspot->x0Error() : 0.0;
}

double FWBeamSpot::y0Error() const
{
   return m_beamspot ? m_beamspot->y0Error() : 0.0;
}

double FWBeamSpot::z0Error() const
{
   return m_beamspot ? m_beamspot->z0Error() : 0.0;
}
