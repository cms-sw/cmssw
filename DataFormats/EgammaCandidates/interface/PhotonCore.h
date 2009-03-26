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
 * \version $Id: PhotonCore.h,v 1.1 2009/03/24 17:59:18 nancy Exp $
 * $Log $
 */
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"


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
    /// add  single ConversionRef to the vector of Refs
    void addConversion( const reco::ConversionRef & r ) { conversions_.push_back(r); }
    /// set electron pixel seed ref
    void addElectronPixelSeed( const reco::ElectronSeedRef & r ) { electronSeed_.push_back(r) ; }

    /// get reference to SuperCluster
    reco::SuperClusterRef superCluster() const {return superCluster_;}
    /// get vector of references to  Conversion's
    reco::ConversionRefVector conversions() const {return conversions_;} 
    /// get reference to electron seed if existing
    reco::ElectronSeedRefVector electronPixelSeeds() const {return electronSeed_;}
 

  private:

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    // vector of references to Conversions
    reco::ConversionRefVector  conversions_;
    // vector of references to ElectronPixelSeeds
    reco::ElectronSeedRefVector  electronSeed_;
   


  };

  
}

#endif
