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
// $Id$
//


#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Point3D.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"



class EgammaHLTTrackIsolation
{

 public:


  EgammaHLTTrackIsolation(float egTrkIso_Electron_PtMin        = 1.5,
			  float egTrkIso_Electron_ConeSize     = 0.2,
			  float egTrkIso_Electron_ZSpan        = 0.1,
			  float egTrkIso_Electron_RSpan        = 99999,
			  float egTrkIso_Electron_VetoConeSize = 0.02,
			  float egTrkIso_Photon_PtMin          = 1.5,
			  float egTrkIso_Photon_ConeSize       = 0.3,
			  float egTrkIso_Photon_ZSpan          = 99999,
			  float egTrkIso_Photon_RSpan          = 99999){
    
    ptMin        = egTrkIso_Electron_PtMin;
    conesize     = egTrkIso_Electron_ConeSize;
    zspan        = egTrkIso_Electron_ZSpan;
    rspan        = egTrkIso_Electron_RSpan;
    vetoConesize = egTrkIso_Electron_VetoConeSize;
    ptMinG       = egTrkIso_Photon_PtMin;
    conesizeG    = egTrkIso_Photon_ConeSize;
    zspanG       = egTrkIso_Photon_ZSpan; 
    rspanG       = egTrkIso_Photon_RSpan;

    /*
    edm::LogInfo ("category") << "EgammaHLTTrackIsolation instance:"
			      << " ptMin=" << ptMin << "|" << ptMinG
			      << " conesize="<< conesize << "|" << conesizeG
			      << " zspan=" << zspan << "|" << zspanG
			      << " rspan=" << rspan << "|" << rspanG 
			      << " vetoConesize="<< vetoConesize
			      << std::endl;    
    */    
  }



  virtual ~EgammaHLTTrackIsolation();


  /// Get number of tracks and Pt sum of tracks inside an isolation cone for electrons
  std::pair<int,float> electronIsolation(const reco::Track * const tr, const reco::TrackCollection& isoTracks);
  std::pair<int,float> electronIsolation(const reco::Track * const tr, const reco::TrackCollection& isoTracks, GlobalPoint vertex);
  
  /// Get number of tracks and Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use PhotonCandidate vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  std::pair<int,float> photonIsolation(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, bool useVertex);
  std::pair<int,float> photonIsolation(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, GlobalPoint vertex);

  /// Get number of tracks inside an isolation cone for electrons
  int electronTrackCount(const reco::Track * const tr, const reco::TrackCollection& isoTracks)
  {return electronIsolation(tr,isoTracks).first;}
  int electronTrackCount(const reco::Track * const tr, const reco::TrackCollection& isoTracks, GlobalPoint vertex)
  {return electronIsolation(tr,isoTracks,vertex).first;}

  /// Get number of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  int photonTrackCount(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, bool useVertex)
  {return photonIsolation(pho,isoTracks,useVertex).first;}
  int photonTrackCount(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, GlobalPoint vertex)
  {return photonIsolation(pho,isoTracks,vertex).first;}

  /// Get Pt sum of tracks inside an isolation cone for electrons
  float electronPtSum(const reco::Track * const tr, const reco::TrackCollection& isoTracks)
  {return electronIsolation(tr,isoTracks).second;}
  float electronPtSum(const reco::Track * const tr, const reco::TrackCollection& isoTracks, GlobalPoint vertex)
  {return electronIsolation(tr,isoTracks,vertex).second;}

  /// Get Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  float photonPtSum(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, bool useVertex)
  {return photonIsolation(pho,isoTracks, useVertex).second;}
  float photonPtSum(const reco::Photon * const pho, const reco::TrackCollection& isoTracks, GlobalPoint vertex)
  {return photonIsolation(pho,isoTracks, vertex).second;}


  /// Get pt cut for itracks.
  float getPtMin(bool getE=true) { 
    if(getE) return ptMin; 
    else return ptMinG; }
  /// Get isolation cone size. 
  float getConeSize(bool getE=true) { 
    if(getE) return conesize; 
    else return conesizeG; }
  /// Get maximum ivertex z-coordinate spread.
  float getZspan(bool getE=true) {
    if(getE) return zspan; 
    else return zspanG; }
  /// Get maximum transverse distance of ivertex from beam line.
  float getRspan(bool getE=true) { 
    if(getE) return rspan; 
    else return rspanG; }



   private:

  // Call track reconstruction
  std::pair<int,float> findIsoTracks(GlobalVector mom, GlobalPoint vtx, const reco::TrackCollection& isoTracks, bool isElectron, bool useVertex=true);

  
  // Protected instance of the class itself
  //static EgammaL3TrackIsolation *theinstance;

  // Parameters of isolation cone geometry.
  // I Electron case
  float ptMin;
  float conesize;
  float zspan;
  float rspan;
  float vetoConesize;
  // II Photon case (G for Gamma)
  float ptMinG;
  float conesizeG;
  float zspanG;
  float rspanG;



};


#endif
