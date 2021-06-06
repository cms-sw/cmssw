#ifndef CalibrationIsolatedParticlesCaloPropagateTrack_h
#define CalibrationIsolatedParticlesCaloPropagateTrack_h

#include <cmath>
#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//sim track
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

namespace spr {

  struct propagatedTrack {
    propagatedTrack() { ok = false; }
    bool ok;
    math::XYZPoint point;
    GlobalVector direction;
  };

  struct propagatedTrackID {
    propagatedTrackID() {
      ok = false;
      okECAL = false;
      okHCAL = false;
    }
    bool ok, okECAL, okHCAL;
    DetId detIdECAL, detIdHCAL, detIdEHCAL;
    double etaECAL, etaHCAL;
    double phiECAL, phiHCAL;
    reco::TrackCollection::const_iterator trkItr;
  };

  struct propagatedTrackDirection {
    propagatedTrackDirection() {
      ok = false;
      okECAL = false;
      okHCAL = false;
    }
    bool ok, okECAL, okHCAL;
    DetId detIdECAL, detIdHCAL, detIdEHCAL;
    GlobalPoint pointECAL, pointHCAL;
    GlobalVector directionECAL, directionHCAL;
    reco::TrackCollection::const_iterator trkItr;
  };

  struct propagatedGenTrackID {
    propagatedGenTrackID() {
      ok = okECAL = okHCAL = false;
      charge = pdgId = 0;
    }
    bool ok, okECAL, okHCAL;
    DetId detIdECAL, detIdHCAL, detIdEHCAL;
    GlobalPoint pointECAL, pointHCAL;
    GlobalVector directionECAL, directionHCAL;
    int charge, pdgId;
    HepMC::GenEvent::particle_const_iterator trkItr;
  };

  struct propagatedGenParticleID {
    propagatedGenParticleID() {
      ok = okECAL = okHCAL = false;
      charge = pdgId = 0;
    }
    bool ok, okECAL, okHCAL;
    DetId detIdECAL, detIdHCAL, detIdEHCAL;
    GlobalPoint pointECAL, pointHCAL;
    GlobalVector directionECAL, directionHCAL;
    int charge, pdgId;
    reco::GenParticleCollection::const_iterator trkItr;
  };

  struct trackAtOrigin {
    trackAtOrigin() { ok = false; }
    bool ok;
    int charge;
    GlobalPoint position;
    GlobalVector momentum;
  };

  // Returns a vector of DetID's of closest cell on the ECAL/HCAL surface of
  // all the tracks in the collection. Also saves a boolean if extrapolation
  // is satisfactory
  std::vector<spr::propagatedTrackID> propagateCosmicCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                                                          const CaloGeometry* geo,
                                                          const MagneticField* bField,
                                                          const std::string& theTrackQuality,
                                                          bool debug = false);
  std::vector<spr::propagatedTrackID> propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                                                    const CaloGeometry* geo,
                                                    const MagneticField* bField,
                                                    const std::string& theTrackQuality,
                                                    bool debug = false);
  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     const std::string& theTrackQuality,
                     std::vector<spr::propagatedTrackID>& vdets,
                     bool debug = false);
  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     const std::string& theTrackQuality,
                     std::vector<spr::propagatedTrackDirection>& trkDir,
                     bool debug = false);
  spr::propagatedTrackID propagateCALO(const reco::Track*,
                                       const CaloGeometry* geo,
                                       const MagneticField* bField,
                                       bool debug = false);
  std::vector<spr::propagatedGenTrackID> propagateCALO(const HepMC::GenEvent* genEvent,
                                                       const ParticleDataTable* pdt,
                                                       const CaloGeometry* geo,
                                                       const MagneticField* bField,
                                                       double etaMax = 3.0,
                                                       bool debug = false);
  std::vector<spr::propagatedGenParticleID> propagateCALO(edm::Handle<reco::GenParticleCollection>& genParticles,
                                                          const ParticleDataTable* pdt,
                                                          const CaloGeometry* geo,
                                                          const MagneticField* bField,
                                                          double etaMax = 3.0,
                                                          bool debug = false);
  spr::propagatedTrackDirection propagateCALO(unsigned int thisTrk,
                                              edm::Handle<edm::SimTrackContainer>& SimTk,
                                              edm::Handle<edm::SimVertexContainer>& SimVtx,
                                              const CaloGeometry* geo,
                                              const MagneticField* bField,
                                              bool debug = false);
  spr::propagatedTrackDirection propagateHCALBack(unsigned int thisTrk,
                                                  edm::Handle<edm::SimTrackContainer>& SimTk,
                                                  edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                  const CaloGeometry* geo,
                                                  const MagneticField* bField,
                                                  bool debug = false);
  std::pair<bool, HcalDetId> propagateHCALBack(const reco::Track*,
                                               const CaloGeometry* geo,
                                               const MagneticField* bField,
                                               bool debug = false);

  // Propagate tracks to the ECAL surface and optionally returns the
  // extrapolated point (and the track direction at point of extrapolation)
  spr::propagatedTrack propagateTrackToECAL(const reco::Track*, const MagneticField*, bool debug = false);
  spr::propagatedTrack propagateTrackToECAL(unsigned int thisTrk,
                                            edm::Handle<edm::SimTrackContainer>& SimTk,
                                            edm::Handle<edm::SimVertexContainer>& SimVtx,
                                            const MagneticField*,
                                            bool debug = false);
  std::pair<math::XYZPoint, bool> propagateECAL(const reco::Track*, const MagneticField*, bool debug = false);
  std::pair<DetId, bool> propagateIdECAL(const HcalDetId& id,
                                         const CaloGeometry* geo,
                                         const MagneticField*,
                                         bool debug = false);
  std::pair<math::XYZPoint, bool> propagateECAL(
      const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug = false);

  // Propagate tracks to the HCAL surface and optionally returns the
  // extrapolated point (and the track direction at point of extrapolation)
  spr::propagatedTrack propagateTrackToHCAL(const reco::Track*, const MagneticField*, bool debug = false);
  spr::propagatedTrack propagateTrackToHCAL(unsigned int thisTrk,
                                            edm::Handle<edm::SimTrackContainer>& SimTk,
                                            edm::Handle<edm::SimVertexContainer>& SimVtx,
                                            const MagneticField*,
                                            bool debug = false);
  std::pair<math::XYZPoint, bool> propagateHCAL(const reco::Track*, const MagneticField*, bool debug = false);
  std::pair<math::XYZPoint, bool> propagateHCAL(
      const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug = false);

  // Propagate the track to the end of the tracker and returns the extrapolated
  // point and optionally the length of the track upto the end
  std::pair<math::XYZPoint, bool> propagateTracker(const reco::Track*, const MagneticField*, bool debug = false);
  std::pair<math::XYZPoint, double> propagateTrackerEnd(const reco::Track*, const MagneticField*, bool debug = false);

  spr::propagatedTrack propagateCalo(const GlobalPoint& vertex,
                                     const GlobalVector& momentum,
                                     int charge,
                                     const MagneticField*,
                                     float zdist,
                                     float radius,
                                     float corner,
                                     bool debug = false);

  // Gives the vertex and momentum of a SimTrack
  spr::trackAtOrigin simTrackAtOrigin(unsigned int thisTrk,
                                      edm::Handle<edm::SimTrackContainer>& SimTk,
                                      edm::Handle<edm::SimVertexContainer>& SimVtx,
                                      bool debug = false);

  //Get HcalDetID's for two values of r/z
  bool propagateHCAL(const reco::Track* track,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     bool typeRZ,
                     const std::pair<double, double> rz,
                     bool debug);
  bool propagateHCAL(unsigned int thisTrk,
                     edm::Handle<edm::SimTrackContainer>& SimTk,
                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     bool typeRZ,
                     const std::pair<double, double> rz,
                     bool debug);
  std::pair<HcalDetId, HcalDetId> propagateHCAL(const CaloGeometry* geo,
                                                const MagneticField* bField,
                                                const GlobalPoint& vertex,
                                                const GlobalVector& momentum,
                                                int charge,
                                                bool typeRZ,
                                                const std::pair<double, double> rz,
                                                bool debug);

}  // namespace spr
#endif
