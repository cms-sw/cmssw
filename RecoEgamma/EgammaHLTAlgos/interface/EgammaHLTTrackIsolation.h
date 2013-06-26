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
// $Id: EgammaHLTTrackIsolation.h,v 1.6 2010/08/12 15:25:02 sharper Exp $
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



class EgammaHLTTrackIsolation
{

 public:


  EgammaHLTTrackIsolation(double egTrkIso_PtMin, 
			  double egTrkIso_ConeSize,
			  double egTrkIso_ZSpan,   
			  double egTrkIso_RSpan,  
			  double egTrkIso_VetoConeSize,
			  double egTrkIso_stripBarrel=0,
			  double egTrkIso_stripEndcap=0) :
    ptMin(egTrkIso_PtMin),
    conesize(egTrkIso_ConeSize),
    zspan(egTrkIso_ZSpan),
    rspan(egTrkIso_RSpan),
    vetoConesize(egTrkIso_VetoConeSize),
    stripBarrel(egTrkIso_stripBarrel),
    stripEndcap(egTrkIso_stripEndcap)
  {
      
      /*
	std::cout << "EgammaHLTTrackIsolation instance:"
	<< " ptMin=" << ptMin << " "
	<< " conesize="<< conesize << " "
	<< " zspan=" << zspan << " "
	<< " rspan=" << rspan << " "
	<< " vetoConesize="<< vetoConesize
	<< std::endl;    
      */
    }


  /// Get number of tracks and Pt sum of tracks inside an isolation cone for electrons
  std::pair<int,float> electronIsolation(const reco::Track * const tr, const reco::TrackCollection* isoTracks);
  std::pair<int,float> electronIsolation(const reco::Track * const tr, const reco::ElectronCollection* allEle, const reco::TrackCollection* isoTracks);
  std::pair<int,float> electronIsolation(const reco::Track * const tr, const reco::TrackCollection* isoTracks, GlobalPoint vertex);
  
  /// Get number of tracks and Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use PhotonCandidate vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  std::pair<int,float> photonIsolation(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, bool useVertex);
  std::pair<int,float> photonIsolation(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, GlobalPoint vertex);
  std::pair<int,float> photonIsolation(const reco::RecoCandidate * const recocand, const reco::ElectronCollection* allEle, const reco::TrackCollection* isoTracks);  

  /// Get number of tracks inside an isolation cone for electrons
  int electronTrackCount(const reco::Track * const tr, const reco::TrackCollection* isoTracks)
  {return electronIsolation(tr,isoTracks).first;}
  int electronTrackCount(const reco::Track * const tr, const reco::TrackCollection* isoTracks, GlobalPoint vertex)
  {return electronIsolation(tr,isoTracks,vertex).first;}

  /// Get number of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  int photonTrackCount(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, bool useVertex)
  {return photonIsolation(recocand,isoTracks,useVertex).first;}
  int photonTrackCount(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, GlobalPoint vertex)
  {return photonIsolation(recocand,isoTracks,vertex).first;}
  int photonTrackCount(const reco::RecoCandidate * const recocand, const reco::ElectronCollection* allEle, const reco::TrackCollection* isoTracks)
  {return photonIsolation(recocand,allEle,isoTracks).first;}

  /// Get Pt sum of tracks inside an isolation cone for electrons
  float electronPtSum(const reco::Track * const tr, const reco::TrackCollection* isoTracks)
  {return electronIsolation(tr,isoTracks).second;}
  float electronPtSum(const reco::Track * const tr, const reco::TrackCollection* isoTracks, GlobalPoint vertex)
  {return electronIsolation(tr,isoTracks,vertex).second;}
  float electronPtSum(const reco::Track * const tr, const reco::ElectronCollection* allEle ,const reco::TrackCollection* isoTracks)
  {return electronIsolation(tr,allEle,isoTracks).second;} 

  /// Get Pt sum of tracks inside an isolation cone for photons
  /// set useVertex=true to use Photon vertex from EgammaPhotonVtxFinder
  /// set useVertex=false to consider all tracks for isolation
  float photonPtSum(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, bool useVertex)
  {return photonIsolation(recocand,isoTracks, useVertex).second;}
  float photonPtSum(const reco::RecoCandidate * const recocand, const reco::TrackCollection* isoTracks, GlobalPoint vertex)
  {return photonIsolation(recocand,isoTracks, vertex).second;}
  float photonPtSum(const reco::RecoCandidate * const recocand, const reco::ElectronCollection* allEle, const reco::TrackCollection* isoTracks)
  {return photonIsolation(recocand,allEle,isoTracks).second;}


  /// Get pt cut for itracks.
  double getPtMin() { return ptMin;}
  /// Get isolation cone size. 
  double getConeSize() { return conesize; }
  /// Get maximum ivertex z-coordinate spread.
  double getZspan() {return zspan; }
  /// Get maximum transverse distance of ivertex from beam line.
  double getRspan() { return rspan; }
  /// Get veto cone size
  double getvetoConesize() { return vetoConesize; }

   private:
  // Call track reconstruction
  std::pair<int,float> findIsoTracks(GlobalVector mom, GlobalPoint vtx, const reco::TrackCollection* isoTracks, bool isElectron, bool useVertex=true);
  std::pair<int,float> findIsoTracksWithoutEle(GlobalVector mom, GlobalPoint vtx, const reco::ElectronCollection* allEle, const reco::TrackCollection* isoTracks);

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
