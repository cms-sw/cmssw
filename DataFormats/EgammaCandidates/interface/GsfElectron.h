#ifndef GsfElectron_h
#define GsfElectron_h
/** \class reco::Electron
 *
 * An Electron with GsfTrack seeded from an ElectronPixelSeed
 * adapted from the TRecElectron class in ORCA
 *
 * \author U.Berthon, ClaudeCharlot, LLR
 *
 * \version $Id: GsfElectron.h,v 1.13 2008/12/03 18:00:32 charlot Exp $
 *
 */

//-------------------------------------------------------------------
//
// Package EgammaCandidates
//
/** \class GsfElectron
*/
// Offline electrons with Guassian Sum Filter tracking.
// Renamed from PixelMatchGsfElectron.
//
// Author:
//
// Claude Charlot - CNRS & IN2P3, LLR Ecole polytechnique
// Ursula Berthon - LLR Ecole polytechnique
//
// $Log: GsfElectron.h,v $
// Revision 1.13  2008/12/03 18:00:32  charlot
// add identification of electron cluster and associated matching variables
//
// Revision 1.12  2008/12/01 13:03:00  chamont
// store ambiguous gsf track into electrons
//
// Revision 1.11  2008/10/21 12:57:56  chamont
// use infinity as a defaut value for new attributes scSigmaEtaEta and scSigmaIEtaIEta, and code cleaning
//
// Revision 1.10  2008/10/20 12:25:00  chamont
// few doxygen comments
//
// Revision 1.9  2008/10/17 13:41:50  chamont
// new attributes for cluster shape and best fitting ctf track
//
// Revision 1.8  2008/09/18 08:08:23  charlot
// updated description of classification
//
// Revision 1.7  2008/04/21 14:05:23  llista
// added virtual function to identify particle type
//
// Revision 1.6  2008/04/10 08:45:06  uberthon
// remove ClusterShape from GsfElectron, remove obsolete classes PixelMatchElectron, PixelMatchGsfElectron
//
// Revision 1.5  2007/12/11 16:22:02  uberthon
// remove annoying LogWarning
//
// Revision 1.4  2007/12/11 10:35:47  futyand
// revert HEAD to 18X version
//
// Revision 1.2  2007/12/10 21:01:16  futyand
// add typedefs to allow PixelMatchGsfElectron and GsfElectron to be used interchangably for a transition period
//
// Revision 1.1  2007/12/08 13:07:50  futyand
// Renamed from PixelMatchGsfElectron
//
//
//-------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


#include <vector>
#include <limits>

namespace reco {

class GsfElectron : public RecoCandidate {

 public:

  GsfElectron() ;

  //! one must give almost all attributes values when creating an electron
  GsfElectron(
	const LorentzVector & p4,
	const SuperClusterRef scl,
	const GsfTrackRef gsfTrack,
	const GlobalPoint & tssuperPos, const GlobalVector & tssuperMom,
	const GlobalPoint & tsseedPos, const GlobalVector & tsseedMom,
	const GlobalPoint & innPos, const GlobalVector & innMom,
	const GlobalPoint & vtxPos, const GlobalVector & vtxMom,
	const GlobalPoint & outPos, const GlobalVector & outMom,
	double HoE,
	float scSigmaEtaEta =std::numeric_limits<float>::infinity(),
	float scSigmaIEtaIEta =std::numeric_limits<float>::infinity(),
	float scE1x5 =0., float scE2x5Max =0., float scE5x5 =0.,
        const TrackRef ctfTrack =TrackRef(), const float shFracInnerHits =0.,
	const BasicClusterRef electronCluster=BasicClusterRef(),
	const GlobalPoint & tselePos=GlobalPoint(), const GlobalVector & tseleMom=GlobalVector()
	) ;

  virtual ~GsfElectron(){};

  //Public methods

  // particle behaviour
   /** The electron classification.
      barrel  :   0: golden,  10: bigbrem,  20: narrow, 30-34: showering,
                (30: showering nbrem=0, 31: showering nbrem=1, 32: showering nbrem=2 ,33: showering nbrem=3, 34: showering nbrem>=4)
                 40: crack, 41: eta gaps, 42: phi gaps
      endcaps : 100: golden, 110: bigbrem, 120: narrow, 130-134: showering
               (130: showering nbrem=0, 131: showering nbrem=1, 132: showering nbrem=2 ,133: showering nbrem=3, 134: showering nbrem>=4)
                140: crack
   */
  int classification() const {return electronClass_;}

  GsfElectron * clone() const;

  // setters
  void setDeltaEtaSuperClusterAtVtx(float de) {deltaEtaSuperClusterAtVtx_=de;}
  void setDeltaPhiSuperClusterAtVtx (float dphi) {deltaPhiSuperClusterAtVtx_=dphi;}

  void setSuperCluster(const reco::SuperClusterRef &scl) // used by GsfElectronSelector.h
    { superCluster_=scl ; }
  void setGsfTrack( const reco::GsfTrackRef & t )
   { track_=t ; }
  void addAmbiguousGsfTrack( const reco::GsfTrackRef & t )
   { ambiguousGsfTracks_.push_back(t) ; }

  // supercluster and electron track related quantities
  //! the super cluster energy corrected by EnergyScaleFactor
  float caloEnergy() const {return superClusterEnergy_;}
  //! the super cluster position
  math::XYZPoint caloPosition() const {return superCluster()->position();}
  //! the track momentum at vertex
  math::XYZVector trackMomentumAtVtx() const {return trackMomentumAtVtx_;}
  //! the track impact point state position
  math::XYZVector TrackPositionAtVtx() const {return trackPositionAtVtx_;}
  //! the track momentum extrapolated at the supercluster position
  math::XYZVector trackMomentumAtCalo() const {return trackMomentumAtCalo_;}
  //! the track momentum extrapolated from outermost position at the seed cluster position
  math::XYZVector trackMomentumOut() const {return trackMomentumOut_;}
  //! the track momentum extrapolated from outermost position at the ele cluster position
  math::XYZVector trackMomentumAtEleClus() const {return trackMomentumAtEleClus_;}
  //! the track extrapolated position at min distance to the supercluster position
  math::XYZVector TrackPositionAtCalo() const {return trackPositionAtCalo_;}
  //! the supercluster energy / track momentum at impact point
  float eSuperClusterOverP() const {return eSuperClusterOverP_;}
  //! the seed cluster energy / track momentum at calo from outermost state
  float eSeedClusterOverPout() const {return eSeedClusterOverPout_;}
  //! the electron cluster energy / track momentum at calo from outermost state
  float eEleClusterOverPout() const {return eEleClusterOverPout_;}
  //! the supercluster eta - track eta from helix extrapolation from impact point
  float deltaEtaSuperClusterTrackAtVtx() const {return deltaEtaSuperClusterAtVtx_;}
  //! the seed cluster eta - track eta at calo from outermost state
  float deltaEtaSeedClusterTrackAtCalo() const {return deltaEtaSeedClusterAtCalo_;}
  //! the electron cluster eta - track eta at calo from outermost state
  float deltaEtaEleClusterTrackAtCalo() const {return deltaEtaEleClusterAtCalo_;}
  //! the supercluster phi - track phi from helix extrapolation from impact point
  float deltaPhiSuperClusterTrackAtVtx() const {return deltaPhiSuperClusterAtVtx_;}
  //! the seed cluster phi - track phi at calo from outermost state
  float deltaPhiSeedClusterTrackAtCalo() const {return deltaPhiSeedClusterAtCalo_;}
  //! the electron cluster phi - track phi at calo from outermost state
  float deltaPhiEleClusterTrackAtCalo() const {return deltaPhiEleClusterAtCalo_;}

  //! the hadronic over electromagnetic fraction
  float hadronicOverEm() const {return hadOverEm_;}

  // corrections
  //! tell if class dependant escale correction have been applied
  bool isEnergyScaleCorrected() const {return energyScaleCorrected_;}
  //! tell if class dependant E-p combination has been determined
  bool isMomentumCorrected() const {return momentumFromEpCombination_;}
  //! handle electron energy correction.  Rescales 4 momentum from corrected
  //! energy value and sets momentumFromEpCombination_ to true
  void correctElectronFourMomentum(const math::XYZTLorentzVectorD & momentum,float & enErr, float  & tMerr);
  //! handle electron supercluster energy scale correction.  Propagates new
  //! energy value to all electron attributes and sets energyScaleCorrected_ to true
  void correctElectronEnergyScale(const float newEnergy);

  //! the brem fraction: (track momentum in - track momentum out) / track momentum in 
  float fbrem() const {return fbrem_;}

  //! determine the class of the electron
  void classifyElectron(const int myclass);

  //! the error on the supercluster energy
  float caloEnergyError() const {return energyError_;}
  //! the error on the supercluster track momentum
  float trackMomentumError() const {return trackMomentumError_;}

  //! get associated superCluster Pointer
  SuperClusterRef superCluster() const { return superCluster_; }

  //! get associated GsfTrack pointer
  reco::GsfTrackRef gsfTrack() const { return track_ ; }
  //! get the CTF track best matching the GTF associated to this electron
  reco::TrackRef track() const { return ctfTrack_ ; }
  //! measure the fraction of common hits between the GSF and CTF tracks
  float shFracInnerHits() const { return shFracInnerHits_ ; }

  //! number of related brem clusters
  int numberOfClusters() const {return superCluster_->clustersSize();}

  //! array of pointers to the related brem clusters
  basicCluster_iterator basicClustersBegin() const { return superCluster_->clustersBegin(); }
  basicCluster_iterator basicClustersEnd() const { return superCluster_->clustersEnd(); }

  bool isElectron() const;

  //! a characteristic from the associated super-cluster
  float scSigmaEtaEta() const { return scSigmaEtaEta_ ; }
  //! a characteristic from the associated super-cluster
  float scSigmaIEtaIEta() const { return scSigmaIEtaIEta_ ; }
  //! a characteristic from the associated super-cluster
  float scE1x5() const { return scE1x5_ ; }
  //! a characteristic from the associated super-cluster
  float scE2x5Max() const { return scE2x5Max_ ; }
  //! a characteristic from the associated super-cluster
  float scE5x5() const { return scE5x5_ ; }

  //! accessor to the ambiguous gsf tracks
  GsfTrackRefVector::size_type ambiguousGsfTracksSize() const
   { return ambiguousGsfTracks_.size() ; }
  //! accessor to the ambiguous gsf tracks
  GsfTrackRefVector::const_iterator ambiguousGsfTracksBegin() const
   { return ambiguousGsfTracks_.begin() ; }
  //! accessor to the ambiguous gsf tracks
  GsfTrackRefVector::const_iterator ambiguousGsfTracksEnd() const
   { return ambiguousGsfTracks_.end() ; }
   
  //! access to the electron basic cluster
  BasicClusterRef electronCluster() const { return electronCluster_; }
   

private:

  // temporary
  //  float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
  //  float ecalPhi(float PtParticle, float EtaParticle, float PhiParticle, int ChargeParticle, float Rstart);

  math::XYZVector trackMomentumAtVtx_;
  math::XYZVector trackPositionAtVtx_;
  math::XYZVector trackMomentumAtCalo_;
  math::XYZVector trackPositionAtCalo_;
  math::XYZVector trackMomentumOut_;

  float energyError_;
  float trackMomentumError_;

  int electronClass_;

  float superClusterEnergy_;
  float eSuperClusterOverP_;
  float eSeedClusterOverPout_;

  float deltaEtaSuperClusterAtVtx_;
  float deltaEtaSeedClusterAtCalo_;
  float deltaPhiSuperClusterAtVtx_;
  float deltaPhiSeedClusterAtCalo_;

  float hadOverEm_;

  reco::SuperClusterRef superCluster_;
  reco::GsfTrackRef track_;

  bool energyScaleCorrected_;
  bool momentumFromEpCombination_;

  // super-cluster characteristics
  float scSigmaEtaEta_ ;
  float scSigmaIEtaIEta_ ;
  float scE1x5_ ;
  float scE2x5Max_ ;
  float scE5x5_ ;

  // ctf track
  reco::TrackRef ctfTrack_;
  float shFracInnerHits_;

  // ambiguous gsf tracks
  reco::GsfTrackRefVector ambiguousGsfTracks_ ;

  reco::BasicClusterRef electronCluster_;

  math::XYZVector trackMomentumAtEleClus_;

  float eEleClusterOverPout_;
  float deltaEtaEleClusterAtCalo_;
  float deltaPhiEleClusterAtCalo_;

  // brem fraction
  float fbrem_;
  
  /// check overlap with another candidate
  virtual bool overlap( const Candidate & ) const;

};

  typedef GsfElectron PixelMatchGsfElectron ;

}
#endif
