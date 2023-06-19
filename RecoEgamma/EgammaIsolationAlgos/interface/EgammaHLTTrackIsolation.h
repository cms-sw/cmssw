#ifndef EgammaHLTAlgos_EgammaHLTTrackIsolation_h
#define EgammaHLTAlgos_EgammaHLTTrackIsolation_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTTrackIsolation
//
/**\class EgammaHLTTrackIsolation EgammaHLTTrackIsolation.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h

 Description: Number of tracks inside an isolation cone, with con geometry defined by ptMin, conesize, rspan and zspan. 
 Usage:
    <usage>

*/
//
// Original Author:  Monica Vazquez Acosta - CERN
//         Created:  Tue Jun 13 12:19:32 CEST 2006
//

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaHLTTrackIsolation {
public:
  EgammaHLTTrackIsolation(const edm::ParameterSet& iConfig)
      : ptMin(iConfig.getParameter<double>("ptMin")),
        conesize(iConfig.getParameter<double>("coneSize")),
        zspan(iConfig.getParameter<double>("zspan")),
        rspan(iConfig.getParameter<double>("rspan")),
        vetoConesize(iConfig.getParameter<double>("vetoConeSize")),
        stripBarrel(iConfig.getParameter<double>("stripBarrel")),
        stripEndcap(iConfig.getParameter<double>("stripEndcap")) {}
  EgammaHLTTrackIsolation(double egTrkIso_PtMin,
                          double egTrkIso_ConeSize,
                          double egTrkIso_ZSpan,
                          double egTrkIso_RSpan,
                          double egTrkIso_VetoConeSize,
                          double egTrkIso_stripBarrel = 0,
                          double egTrkIso_stripEndcap = 0)
      : ptMin(egTrkIso_PtMin),
        conesize(egTrkIso_ConeSize),
        zspan(egTrkIso_ZSpan),
        rspan(egTrkIso_RSpan),
        vetoConesize(egTrkIso_VetoConeSize),
        stripBarrel(egTrkIso_stripBarrel),
        stripEndcap(egTrkIso_stripEndcap) {}

  /// Get number of tracks and Pt sum of tracks inside an isolation cone for electrons
  std::pair<int, float> electronIsolation(const reco::Track* const tr, const reco::TrackCollection* isoTracks) const;
  std::pair<int, float> electronIsolation(const reco::Track* const tr,
                                          const reco::ElectronCollection* allEle,
                                          const reco::TrackCollection* isoTracks) const;
  std::pair<int, float> electronIsolation(const reco::Track* const tr,
                                          const reco::TrackCollection* isoTracks,
                                          GlobalPoint vertex) const;

  /// Get number of tracks and Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use PhotonCandidate vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  std::pair<int, float> photonIsolation(const reco::RecoCandidate* const recocand,
                                        const reco::TrackCollection* isoTracks,
                                        bool useVertex) const;
  std::pair<int, float> photonIsolation(const reco::RecoCandidate* const recocand,
                                        const reco::TrackCollection* isoTracks,
                                        GlobalPoint vertex) const;
  std::pair<int, float> photonIsolation(const reco::RecoCandidate::Point& pos,
                                        const reco::TrackCollection* isoTracks,
                                        GlobalPoint vertex = GlobalPoint(0, 0, 0)) const;

  std::pair<int, float> photonIsolation(const reco::RecoCandidate* const recocand,
                                        const reco::ElectronCollection* allEle,
                                        const reco::TrackCollection* isoTracks) const;

  std::pair<int, float> photonIsolation(float phoEta,
                                        float phiPhi,
                                        const reco::ElectronCollection* allEle,
                                        const reco::TrackCollection* isoTracks) const;

  /// Get number of tracks inside an isolation cone for electrons
  int electronTrackCount(const reco::Track* const tr, const reco::TrackCollection* isoTracks) const {
    return electronIsolation(tr, isoTracks).first;
  }
  int electronTrackCount(const reco::Track* const tr,
                         const reco::TrackCollection* isoTracks,
                         GlobalPoint vertex) const {
    return electronIsolation(tr, isoTracks, vertex).first;
  }

  /// Get number of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  int photonTrackCount(const reco::RecoCandidate* const recocand,
                       const reco::TrackCollection* isoTracks,
                       bool useVertex) const {
    return photonIsolation(recocand, isoTracks, useVertex).first;
  }
  int photonTrackCount(const reco::RecoCandidate* const recocand,
                       const reco::TrackCollection* isoTracks,
                       GlobalPoint vertex) const {
    return photonIsolation(recocand, isoTracks, vertex).first;
  }
  int photonTrackCount(const reco::RecoCandidate* const recocand,
                       const reco::ElectronCollection* allEle,
                       const reco::TrackCollection* isoTracks) const {
    return photonIsolation(recocand, allEle, isoTracks).first;
  }

  /// Get Pt sum of tracks inside an isolation cone for electrons
  float electronPtSum(const reco::Track* const tr, const reco::TrackCollection* isoTracks) const {
    return electronIsolation(tr, isoTracks).second;
  }
  float electronPtSum(const reco::Track* const tr, const reco::TrackCollection* isoTracks, GlobalPoint vertex) const {
    return electronIsolation(tr, isoTracks, vertex).second;
  }
  float electronPtSum(const reco::Track* const tr,
                      const reco::ElectronCollection* allEle,
                      const reco::TrackCollection* isoTracks) const {
    return electronIsolation(tr, allEle, isoTracks).second;
  }

  /// Get Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  float photonPtSum(const reco::RecoCandidate* const recocand,
                    const reco::TrackCollection* isoTracks,
                    bool useVertex) const {
    return photonIsolation(recocand, isoTracks, useVertex).second;
  }
  float photonPtSum(const reco::RecoCandidate* const recocand,
                    const reco::TrackCollection* isoTracks,
                    GlobalPoint vertex) const {
    return photonIsolation(recocand, isoTracks, vertex).second;
  }
  float photonPtSum(const reco::RecoCandidate* const recocand,
                    const reco::ElectronCollection* allEle,
                    const reco::TrackCollection* isoTracks) const {
    return photonIsolation(recocand, allEle, isoTracks).second;
  }

  /// Get pt cut for itracks.
  double getPtMin() const { return ptMin; }
  /// Get isolation cone size.
  double getConeSize() const { return conesize; }
  /// Get maximum ivertex z-coordinate spread.
  double getZspan() const { return zspan; }
  /// Get maximum transverse distance of ivertex from beam line.
  double getRspan() const { return rspan; }
  /// Get veto cone size
  double getvetoConesize() const { return vetoConesize; }

private:
  // Call track reconstruction
  std::pair<int, float> findIsoTracks(GlobalVector mom,
                                      GlobalPoint vtx,
                                      const reco::TrackCollection* isoTracks,
                                      bool isElectron,
                                      bool useVertex = true) const;
  std::pair<int, float> findIsoTracksWithoutEle(GlobalVector mom,
                                                GlobalPoint vtx,
                                                const reco::ElectronCollection* allEle,
                                                const reco::TrackCollection* isoTracks) const {
    return findIsoTracksWithoutEle(mom.eta(), mom.phi(), vtx, allEle, isoTracks);
  }
  std::pair<int, float> findIsoTracksWithoutEle(float centreEta,
                                                float centrePhi,
                                                GlobalPoint vtx,
                                                const reco::ElectronCollection* allEle,
                                                const reco::TrackCollection* isoTracks) const;

  // Parameters of isolation cone geometry.
  double ptMin;
  double conesize;
  double zspan;
  double rspan;
  double vetoConesize;

  //added for inner eta strip veto (I'll keep the violation of CMS naming conventions to be consistant)
  double stripBarrel;
  double stripEndcap;
};

#endif
