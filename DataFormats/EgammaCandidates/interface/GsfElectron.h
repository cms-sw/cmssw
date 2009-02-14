#ifndef GsfElectron_h
#define GsfElectron_h
/** \class reco::Electron
 *
 * An Electron with GsfTrack seeded from an ElectronSeed
 * adapted from the TRecElectron class in ORCA
 *
 * \author U.Berthon, ClaudeCharlot, LLR
 *
 * \version $Id: GsfElectron.h,v 1.19 2009/01/12 15:41:36 chamont Exp $
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
// Revision 1.19  2009/01/12 15:41:36  chamont
// *** empty log message ***
//
// Revision 1.18  2008/12/15 19:49:56  nancy
// Move a variable to avoid warning about initialization order
//
// Revision 1.17  2008/12/13 08:44:53  charlot
// using isolation scheme for H/E
//
// Revision 1.16  2008/12/11 18:13:51  charlot
// updated doxygen comments
//
// Revision 1.15  2008/12/11 17:44:30  charlot
// added ESeedClusterOverP and access method
//
// Revision 1.14  2008/12/05 17:01:13  charlot
// cleaning in extrapolations for matching variables; added fbrem as GsfElectron data member; update of analyzers accordingly
//
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

  //! gsf electron constructor
  GsfElectron(
	const LorentzVector & p4,
	const SuperClusterRef scl,
	const GsfTrackRef gsfTrack,
	const GlobalPoint & tssuperPos, const GlobalVector & tssuperMom,
	const GlobalPoint & tsseedPos, const GlobalVector & tsseedMom,
	const GlobalPoint & innPos, const GlobalVector & innMom,
	const GlobalPoint & vtxPos, const GlobalVector & vtxMom,
	const GlobalPoint & outPos, const GlobalVector & outMom,
	double hadOverEm1, double hadOverEm2,
	float scSigmaEtaEta =std::numeric_limits<float>::infinity(),
	float scSigmaIEtaIEta =std::numeric_limits<float>::infinity(),
	float scE1x5 =0., float scE2x5Max =0., float scE5x5 =0.,
        const TrackRef ctfTrack =TrackRef(), const float shFracInnerHits =0.,
	const BasicClusterRef electronCluster=BasicClusterRef(),
	const GlobalPoint & tselePos=GlobalPoint(), const GlobalVector & tseleMom=GlobalVector()
	) ;

  virtual ~GsfElectron(){};

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

  void setIsEB(bool isEB) {isEB_=true;}
  void setIsEE(bool isEE) {isEE_=true;}
  void setIsEBEEGap(bool isEBEEGap) {isEBEEGap_=true;}
  void setIsEBEtaGap(bool isEBEtaGap) {isEBEtaGap_=true;}
  void setIsEBPhiGap(bool isEBPhiGap) {isEBPhiGap_=true;}
  void setIsEEDeeGap(bool isEEDeeGap) {isEEDeeGap_=true;}
  void setIsEERingGap(bool isEERingGap) {isEERingGap_=true;}
  
  // supercluster and electron track related quantities
  /** the supercluster energy after electron level eta corrections. It differs from the supercluster energy
      only when isEnergyScaleCorrected() returns true.
  */
  float caloEnergy() const {return superClusterEnergy_;}
  //! the super cluster position
  math::XYZPoint caloPosition() const {return superCluster()->position();}
  //! the track momentum at the PCA to the beam spot
  math::XYZVector trackMomentumAtVtx() const {return trackMomentumAtVtx_;}
  //! the track PCA to the beam spot
  math::XYZPoint TrackPositionAtVtx() const {return trackPositionAtVtx_;}
  //! the track momentum extrapolated at the supercluster position from the innermost track state
  math::XYZVector trackMomentumAtCalo() const {return trackMomentumAtCalo_;}
  //! the track momentum extrapolated at the seed cluster position from the outermost track state 
  math::XYZVector trackMomentumOut() const {return trackMomentumOut_;}
  //! the track momentum extrapolated at the ele cluster position from the outermost track state
  math::XYZVector trackMomentumAtEleClus() const {return trackMomentumAtEleClus_;}
  //! the track PCA to the supercluster position
  math::XYZPoint TrackPositionAtCalo() const {return trackPositionAtCalo_;}
  //! the supercluster energy / track momentum at the PCA to the beam spot
  float eSuperClusterOverP() const {return eSuperClusterOverP_;}
  //! the seed cluster energy / track momentum at the PCA to the beam spot
  float eSeedClusterOverP() const {return eSeedClusterOverP_;}
  //! the seed cluster energy / track momentum at calo extrapolated from the outermost track state
  float eSeedClusterOverPout() const {return eSeedClusterOverPout_;}
  //! the electron cluster energy / track momentum at calo extrapolated from the outermost track state
  float eEleClusterOverPout() const {return eEleClusterOverPout_;}
  //! the supercluster eta - track eta position at calo extrapolated from innermost track state
  float deltaEtaSuperClusterTrackAtVtx() const {return deltaEtaSuperClusterAtVtx_;}
  //! the seed cluster eta - track eta position at calo extrapolated from the outermost track state
  float deltaEtaSeedClusterTrackAtCalo() const {return deltaEtaSeedClusterAtCalo_;}
  //! the electron cluster eta - track eta position at calo extrapolated from the outermost state
  float deltaEtaEleClusterTrackAtCalo() const {return deltaEtaEleClusterAtCalo_;}
  //! the supercluster phi - track phi position at calo extrapolated from the innermost track state
  float deltaPhiSuperClusterTrackAtVtx() const {return deltaPhiSuperClusterAtVtx_;}
  //! the seed cluster phi - track phi position at calo extrapolated from the outermost track state
  float deltaPhiSeedClusterTrackAtCalo() const {return deltaPhiSeedClusterAtCalo_;}
  //! the electron cluster phi - track phi position at calo extrapolated from the outermost track state
  float deltaPhiEleClusterTrackAtCalo() const {return deltaPhiEleClusterAtCalo_;}

  //! the hadronic over electromagnetic energy fraction using all hcal depth
  float hadronicOverEm() const {return hadOverEm1_ + hadOverEm2_;}
  //! the hadronic over electromagnetic energy fraction using first hcal depth
  float hadronicOverEm1() const {return hadOverEm1_;}
  //! the hadronic over electromagnetic energy fraction using second hcal depth
  float hadronicOverEm2() const {return hadOverEm2_;}

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

  //! get associated supercluster pointer
  SuperClusterRef superCluster() const { return superCluster_; }

  //! get associated GsfTrack pointer
  reco::GsfTrackRef gsfTrack() const { return track_ ; }
  //! get the CTF track best matching the GTF associated to this electron
  reco::TrackRef track() const { return ctfTrack_ ; }
  //! measure the fraction of common hits between the GSF and CTF tracks
  float shFracInnerHits() const { return shFracInnerHits_ ; }

  //! number of basic clusters inside the supercluster
  int numberOfClusters() const {return superCluster_->clustersSize();}

  //! array of pointers to the related brem clusters
  basicCluster_iterator basicClustersBegin() const { return superCluster_->clustersBegin(); }
  basicCluster_iterator basicClustersEnd() const { return superCluster_->clustersEnd(); }

  bool isElectron() const;

  //! supercluster shape variable
  float scSigmaEtaEta() const { return scSigmaEtaEta_ ; }
  //! supercluster shape variable
  float scSigmaIEtaIEta() const { return scSigmaIEtaIEta_ ; }
  //! supercluster shape variable
  float scE1x5() const { return scE1x5_ ; }
  //! supercluster shape variable
  float scE2x5Max() const { return scE2x5Max_ ; }
  //! supercluster shape variable
  float scE5x5() const { return scE5x5_ ; }
  
  /// Fiducial volume
  //! true if electron is in ECAL barrel
  bool isEB() const{return isEB_;}
  //! true if electron is in ECAL endcap
  bool isEE() const{return isEE_;}
  //! true if electron is in EB, and inside the eta boundaries in super crystals/modules
  bool isEBEtaGap() const{return isEBEtaGap_;}
  //! true if electron is in EB, and inside the phi boundaries in super crystals/modules
  bool isEBPhiGap() const{return isEBPhiGap_;}
  //! true if electron is in EB, and inside the boundaries in super crystals/modules
  bool isEBGap() const{return isEBEtaGap_ || isEBPhiGap_;}
  //! true if electron is in EE, and inside the dee boundaries in supercrystal/D
  bool isEEDeeGap() const{return isEEDeeGap_;}
  //! true if electron is in EE, and inside the ring boundaries in supercrystal/D
  bool isEERingGap() const{return isEERingGap_;}
  //! true if electron is in EE, and inside the boundaries in supercrystal/D
  bool isEEGap() const{return isEEDeeGap_ || isEERingGap_;}
  //! true if electron is in boundary between EB and EE
  bool isEBEEGap() const{return isEBEEGap_;}

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

  math::XYZVector trackMomentumAtVtx_;
  math::XYZPoint trackPositionAtVtx_;
  math::XYZVector trackMomentumAtCalo_;
  math::XYZPoint trackPositionAtCalo_;
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

  // had. over em enrgy using first hcal depth
  float hadOverEm1_;
  // hadronic over em energy using 2nd hcal depth
  float hadOverEm2_;


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
  
  // e seed cluster / pin
  float eSeedClusterOverP_;
  
  // crack and gaps
  bool isEB_;
  bool isEE_;
  bool isEBEtaGap_;
  bool isEBPhiGap_;
  bool isEEDeeGap_;
  bool isEERingGap_;
  bool isEBEEGap_;
  
  /// check overlap with another candidate
  virtual bool overlap( const Candidate & ) const;

};

  typedef GsfElectron PixelMatchGsfElectron ;

}
#endif
