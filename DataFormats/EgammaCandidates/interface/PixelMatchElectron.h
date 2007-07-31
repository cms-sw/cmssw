#ifndef PixelMatchElectron_h
#define PixelMatchElectron_h
//**********************************************************
// For the moment, this is just the interface definition
// Implementation to come
//**********************************************************
//-------------------------------------------------------------------
//
// Package EgammaCandidates
//
/** \class PixelMatchElectron
*/
//  adapted from the TRecElectron class in ORCA
//
// Author:
//
// Claude Charlot - CNRS & IN2P3, LLR Ecole polytechnique
// Ursula Berthon - LLR Ecole polytechnique
// 
// $Log: PixelMatchElectron.h,v $
// Revision 1.7  2007/03/16 12:46:30  uberthon
// make PixelMatchElectrons inherit from RecoCandidate
//
// Revision 1.6  2007/03/13 09:28:37  llista
// updated to latest candidate interface
//
// Revision 1.5  2007/02/26 15:52:56  llista
// restored V00-04-00
//
// Revision 1.3  2007/01/17 10:23:29  llista
// added virtual member function pdgId()
//
// Revision 1.2  2007/01/04 06:35:09  wmtan
// Geometry/Vector moved to DataFormats/GeometryVector
//
// Revision 1.1  2006/12/04 17:47:18  uberthon
// make PixelMatchElectron +PixelMatchGsfElectron separate classes
//
// Revision 1.3  2006/11/14 18:52:22  uberthon
// add some missing data (HoE etc)
//
// Revision 1.2  2006/10/27 15:02:49  uberthon
// add PixelMatchElectron
//
// Revision 1.1  2006/10/18 15:29:56  uberthon
// add PixelMatchElectron class interface
//
// initial version
//
//
//-------------------------------------------------------------------

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


#include <vector>

class ElectronMomentumCorrector;
class ElectronEnergyCorrector;
class ElectronClassification;

namespace reco {
  
  // inheritance of the Electron class is not possible because of the different track type (lack of polymorphism!)
  class PixelMatchElectron : public RecoCandidate {
  public:
    /// default constructor
    PixelMatchElectron() {;}

    /// constructor
    PixelMatchElectron( const SuperClusterRef scl, const TrackRef t,
			const GlobalPoint tssuperPos, const GlobalVector tssuperMom, 
			const GlobalPoint tsseedPos, const GlobalVector tsseedMom, double HoE );
    
    /// destructor
    virtual ~PixelMatchElectron() { }
    
    //Public methods
    PixelMatchElectron * clone() const;
    
    int classification() const { return electronClass_; }
    
    // supercluster and electron track related quantities
    //! the super cluster energy corrected by EnergyScaleFactor
    float caloEnergy() const {return superCluster()->energy();}
    //! the super cluster position
    math::XYZPoint caloPosition() const {return superCluster()->position();}
    //! the track momentum at vertex
    // same as momentum.... math::XYZVector trackMomentumAtVtx() const {return trackMomentumAtVtx_;}
    //! the track impact point state position
    math::XYZVector TrackPositionAtVtx() const {return trackPositionAtVtx_;}
    //! the track momentum extrapolated at the supercluster position
    math::XYZVector trackMomentumAtCalo() const {return trackMomentumAtCalo_;}
    //! the track extrapolated position at min distance to the supercluster position
    math::XYZVector TrackPositionAtCalo() const {return trackPositionAtCalo_;}
    //! the supercluster energy / track momentum at impact point
    float eSuperClusterOverP() const {return eSuperClusterOverP_;}
    //! the seed cluster energy / track momentum at calo from outermost state
    float eSeedClusterOverPout() const {return eSeedClusterOverPout_;}
    //! the supercluster eta - track eta from helix extrapolation from impact point
    float deltaEtaSuperClusterTrackAtVtx() const {return deltaEtaSuperClusterAtVtx_;}
    //! the seed cluster eta - track eta at calo from outermost state
    float deltaEtaSeedClusterTrackAtCalo() const {return deltaEtaSeedClusterAtCalo_;}
    //! the supercluster phi - track phi from helix extrapolation from impact point
    float deltaPhiSuperClusterTrackAtVtx() const {return deltaPhiSuperClusterAtVtx_;}
    //! the seed cluster phi - track phi at calo from outermost state
    float deltaPhiSeedClusterTrackAtCalo() const {return deltaPhiSeedClusterAtCalo_;}
    
    //! the hadronic over electromagnetic fraction
    float hadronicOverEm() const {return hadOverEm_;}
    
    // corrections
    //! tell if class dependant escale correction have been applied
    bool isEnergyScaleCorrected() const {return energyScaleCorrected_;}
    //! tell if class dependant E-p combination has been determined
    bool isMomentumCorrected() const {return momentumFromEpCombination_;}
    //! handle electron energy correction.  Rescales 4 momentum from corrected 
    //! energy value and sets momentumFromEpCombination_ to true
    void correctElectronFourMomentum(const ElectronMomentumCorrector *thecorr) {;}
    //! handle electron supercluster energy scale correction.  Propagates new 
    //! energy value to all electron attributes and sets energyScaleCorrected_ to true
    void correctElectronEnergyScale(const ElectronEnergyCorrector *thecorr) {;}
    //! determine the class of the electron
    void classifyElectron(const ElectronClassification *theclassifier) {;}
    
    //! the errors on the supercluster energy and track momentum
    float caloEnergyError() const {return energyError_;}
    float trackMomentumError() const {return trackMomentumError_;}
    
    //! get associated superCluster Pointer
    SuperClusterRef superCluster() const { return superCluster_; } 

    //! get associated Track pointer
    TrackRef track() const { return track_; } 
    
    //! number of related brem clusters
    int numberOfClusters() const {return superCluster_->clustersSize();}
    
    //! array of pointers to the related brem clusters
    //  BasicClusterRefVector getBremClusters() const;
    basicCluster_iterator basicClustersBegin() const { return superCluster_->clustersBegin(); }
    basicCluster_iterator basicClustersEnd() const { return superCluster_->clustersEnd(); }
    
  private:
    // temporary
    float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
    float ecalPhi(float PtParticle, float EtaParticle, float PhiParticle, int ChargeParticle, float Rstart);
    
    //  math::XYZVector trackMomentumAtVtx_;
    math::XYZVector trackPositionAtVtx_;
    math::XYZVector trackMomentumAtCalo_;
    math::XYZVector trackPositionAtCalo_;
    
    float energyError_;
    float trackMomentumError_;
    
    int electronClass_;
    
    float eSuperClusterOverP_;
    float eSeedClusterOverPout_;
    
    float deltaEtaSuperClusterAtVtx_;
    float deltaEtaSeedClusterAtCalo_;
    float deltaPhiSuperClusterAtVtx_;
    float deltaPhiSeedClusterAtCalo_;
    
    float hadOverEm_;
    
    reco::SuperClusterRef superCluster_;
    //  reco::TrackRef Track_;
    reco::TrackRef track_;
    
    bool energyScaleCorrected_;
    bool momentumFromEpCombination_;

    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
  };
  
}
#endif
