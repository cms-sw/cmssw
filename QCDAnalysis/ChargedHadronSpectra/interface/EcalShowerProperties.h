#ifndef _ChargedHadronSpectra_EcalShowerProperties_h
#define _ChargedHadronSpectra_EcalShowerProperties_h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

namespace edm  { class EventSetup; class Event; }
namespace reco { class Track; }
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class MagneticField;
class Propagator;
class CaloGeometry;
class CaloCellGeometry;

#include <vector>

class EcalShowerProperties
{
  public:
    EcalShowerProperties(const edm::Event & ev, const edm::EventSetup & es);
    std::pair<double,double> processTrack(const reco::Track & track, int & ntime);

  private:
    FreeTrajectoryState getTrajectoryAtOuterPoint(const reco::Track & track);
    Plane::PlanePointer getSurface(const CaloCellGeometry* cell, int i);
    std::vector<TrajectoryStateOnSurface> getEndpoints
     (const FreeTrajectoryState & ftsAtLastPoint,
      const TrajectoryStateOnSurface & tsosBeforeEcal, int subDet);
    double getDistance
     (const std::vector<TrajectoryStateOnSurface> & tsosEnds,
      const CaloCellGeometry* cell);
    std::pair<double,double> processEcalRecHits
      (const std::vector<TrajectoryStateOnSurface> & tsosEnds, int subDet, int & ntime);

    const MagneticField* theMagneticField;
    const Propagator*    thePropagator;
    const CaloGeometry*  theCaloGeometry;

   edm::Handle<EBRecHitCollection>              recHitsBarrel;
   edm::Handle<EERecHitCollection>              recHitsEndcap;
};

#endif
