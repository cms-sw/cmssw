#ifndef CalibrationIsolatedParticlesCaloPropagateTrack_h
#define CalibrationIsolatedParticlesCaloPropagateTrack_h

#include <cmath>
#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

namespace spr{

  struct propagatedTrack {
    propagatedTrack() {ok=false;}
    bool                ok;
    math::XYZPoint      point;
    GlobalVector        direction;
  };

  struct propagatedTrackID {
    propagatedTrackID() {ok=false; okECAL=false; okHCAL=false;}
    bool                                  ok, okECAL, okHCAL;
    DetId                                 detIdECAL, detIdHCAL, detIdEHCAL;
    double                                etaECAL, etaHCAL;
    double                                phiECAL, phiHCAL;
    reco::TrackCollection::const_iterator trkItr;
  };

  struct propagatedTrackDirection {
    propagatedTrackDirection() {ok=false; okECAL=false; okHCAL=false;}
    bool                                  ok, okECAL, okHCAL;
    DetId                                 detIdECAL, detIdHCAL, detIdEHCAL;
    GlobalPoint                           pointECAL, pointHCAL;
    GlobalVector                          directionECAL, directionHCAL;
    reco::TrackCollection::const_iterator trkItr;
  };

  // Returns a vector of DetID's of closest cell on the ECAL/HCAL surface of
  // all the tracks in the collection. Also saves a boolean if extrapolation
  // is satisfactory
  std::vector<spr::propagatedTrackID> propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, bool debug=false);
  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, std::vector<spr::propagatedTrackID>& vdets, bool debug=false);
  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, std::vector<spr::propagatedTrackDirection>& trkDir, bool debug=false);

  // Propagate tracks to the ECAL surface and optionally returns the 
  // extrapolated point (and the track direction at point of extrapolation)
  propagatedTrack propagateTrackToECAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateECAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateECAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug=false);

  // Propagate tracks to the HCAL surface and optionally returns the 
  // extrapolated point (and the track direction at point of extrapolation)
  propagatedTrack propagateTrackToHCAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateHCAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateHCAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug=false);

  // Propagate the track to the end of the tracker and returns the extrapolated
  // point and optionally the length of the track upto the end
  std::pair<math::XYZPoint,bool> propagateTracker(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,double> propagateTrackerEnd(const reco::Track*, const MagneticField*, bool debug=false);

  propagatedTrack propagateCalo(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, float zdist, float radius, float corner, bool debug=false);

}
#endif
