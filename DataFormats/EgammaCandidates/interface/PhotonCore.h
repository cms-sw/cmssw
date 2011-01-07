#ifndef EgammaCandidates_PhotonCore_h
#define EgammaCandidates_PhotonCore_h
/** \class reco::PhotonCore 
 *  
 *  Core description of a Photon. It contains all relevant  
 *  reconstruction information i.e. references to corresponding
 *  SuperCluster, Conversion with its tracks and vertex as well
 *  as to ElectronSeed (if existing for the same SC) 
 *
 *
 * \author  N. Marinelli Univ. of Notre Dame
 * 
 * \version $Id: PhotonCore.h,v 1.2 2009/03/26 12:41:21 nancy Exp $
 * $Log $
 */
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace reco {

  class PhotonCore  {
  public:
    /// default constructor
    PhotonCore() {}

    PhotonCore(const reco::SuperClusterRef & scl ):  superCluster_(scl) { }

    /// destructor
    virtual ~PhotonCore() { }

    PhotonCore* clone() const { return new PhotonCore( * this ); }

    /// set reference to SuperCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set reference to PFlow SuperCluster
    void setPflowSuperCluster( const reco::SuperClusterRef & r ) { pfSuperCluster_ = r; }
    /// add  single ConversionRef to the vector of Refs
    void addConversion( const reco::ConversionRef & r ) { conversions_.push_back(r); }
    /// set electron pixel seed ref
    void addElectronPixelSeed( const reco::ElectronSeedRef & r ) { electronSeed_.push_back(r) ; }
    /// set reference to PFCandidate
    void setPFCandidate( const reco::PFCandidateRef & r ) { pfCandidate_ = r; }


    /// get reference to SuperCluster
    reco::SuperClusterRef superCluster() const {return superCluster_;}
    /// get reference to PFlow SuperCluster
    reco::SuperClusterRef pfSuperCluster() const {return pfSuperCluster_;}
    /// get vector of references to  Conversion's
    reco::ConversionRefVector conversions() const {return conversions_;} 
    /// get reference to electron seed if existing
    reco::ElectronSeedRefVector electronPixelSeeds() const {return electronSeed_;}
    /// get reference to PFlow candidate
    reco::PFCandidateRef pfCandidate() const {return pfCandidate_;}

 

  private:

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    // vector of references to Conversions
    reco::ConversionRefVector  conversions_;
    // vector of references to ElectronPixelSeeds
    reco::ElectronSeedRefVector  electronSeed_;
    /// reference to a Particle flow SuperCluster
    reco::SuperClusterRef pfSuperCluster_;
    /// reference to a Particle flow candidate
    reco::PFCandidateRef pfCandidate_;
   


  };

  
}

#endif
