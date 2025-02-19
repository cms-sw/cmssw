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
 * \version $Id: PhotonCore.h,v 1.6 2011/07/19 16:24:12 nancy Exp $
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
    //    PhotonCore() { }

    /// To be deleted: Internal comment for Florian
    /// I would reserve this constructor to build the standard photons, as it was before, plus I add the initialization of the provenance
    PhotonCore(const reco::SuperClusterRef & scl ):  superCluster_(scl), isPFlowPhoton_(false), isStandardPhoton_(true) { }

    // while for building photons from pf I would use the default constructor 
    PhotonCore(): isPFlowPhoton_(false), isStandardPhoton_(false) { }
    // followed by the setters of the provenance and of the Ref to the wanted supercluster
    // at that point if in PF you have found a photon which correspond to a standard SC yuo can 
    // set both supercluster and the two flags to true
    // if you have found an object which does not have a standard SC associated you set only the
    // one from pflow. 
    // How does this sound ? 


    /// destructor
    virtual ~PhotonCore() { }

    PhotonCore* clone() const { return new PhotonCore( * this ); }

    /// set reference to SuperCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set reference to PFlow SuperCluster
    void setPflowSuperCluster( const reco::SuperClusterRef & r ) { pfSuperCluster_ = r; }
    /// add  single ConversionRef to the vector of Refs
    void addConversion( const reco::ConversionRef & r ) { conversions_.push_back(r); }
    /// add  single ConversionRef to the vector of Refs
    void addOneLegConversion( const reco::ConversionRef & r ) { conversionsOneLeg_.push_back(r); }
    /// set electron pixel seed ref
    void addElectronPixelSeed( const reco::ElectronSeedRef & r ) { electronSeed_.push_back(r) ; }
    /// set the provenance
    void setPFlowPhoton( const bool prov) {  isPFlowPhoton_ = prov; }
    void setStandardPhoton( const bool prov) { isStandardPhoton_ = prov; }


    /// get reference to SuperCluster
    reco::SuperClusterRef superCluster() const { return superCluster_;}
    /// get reference to PFlow SuperCluster
    reco::SuperClusterRef pfSuperCluster() const { return pfSuperCluster_;}

    //// comment for Florian. I have seen that in GsfElectronCore they have a getter for the supercluster
    // which returns the pfSuperCluster only if the standard supeclsuter is not null
    // But I had udnerstood from you when we spke last time that we wish to be free
    // to have both SCs available. Or not ? 

    /// get vector of references to  Conversion's
    reco::ConversionRefVector conversions() const {return conversions_;} 
    /// get vector of references to one leg Conversion's
    reco::ConversionRefVector conversionsOneLeg() const {return conversionsOneLeg_;} 

    /// get reference to electron seed if existing
    reco::ElectronSeedRefVector electronPixelSeeds() const {return electronSeed_;}
    bool isPFlowPhoton() const {return isPFlowPhoton_;} 
    bool isStandardPhoton() const {return isStandardPhoton_;}

  private:

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    // vector of references to Conversions
    reco::ConversionRefVector  conversions_;
    //vector of references for 1-leg
    reco::ConversionRefVector  conversionsOneLeg_;
    // vector of references to ElectronPixelSeeds
    reco::ElectronSeedRefVector  electronSeed_;
    /// reference to a Particle flow SuperCluster
    reco::SuperClusterRef pfSuperCluster_;
    bool  isPFlowPhoton_;
    bool  isStandardPhoton_;

  };

  
}

#endif
