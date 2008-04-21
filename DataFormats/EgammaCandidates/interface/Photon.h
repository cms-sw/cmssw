#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.18 2008/03/03 20:34:30 nancy Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// default constructor
    Photon() : RecoCandidate() { }
    /// constructor from values
    Photon( const LorentzVector & p4, Point unconvPos, 
	    const SuperClusterRef scl, 
            float HoE,
            float r9,
	    float r19,
            float e5x5,
	    bool hasPixelSeed=false, const Point & vtx = Point( 0, 0, 0 ) );
    /// destructor
    virtual ~Photon();
    /// returns a clone of the candidate
    virtual Photon * clone() const;
    /// reference to SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// vector of references to  Conversion's
    std::vector<reco::ConversionRef> conversions() const ; 
     /// set reference to SuperCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// add  single ConversionRef to the vector of Refs
    void addConversion( const reco::ConversionRef & r ) { conversions_.push_back(r); }
    //    void addConversion( const reco::ConvertedPhotonRef & r ) { conversions_.push_back(r); }
       /// set primary event vertex used to define photon direction
    void setVertex(const Point & vertex);
    /// position in ECAL
    math::XYZPoint caloPosition() const;
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint unconvertedPosition() const { return unconvPosition_; }
    /// ratio of E(3x3)/ESC
    float r9() const { return r9_; }
    /// ratio of Emax/E(3x3)
    float r19() const { return  r19_;}
    /// 5x5 energy
    float e5x5() const { return e5x5_ ;}
    //! the hadronic over electromagnetic fraction
    float hadronicOverEm() const {return hadOverEm_;}
    /// Whether or not the SuperCluster has a matched pixel seed
    bool hasPixelSeed() const { return pixelSeed_; }
    /// Bool flagging photons with a vector of refereces to conversions with size >0
    bool isConverted() const;

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint unconvPosition_;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    // vector of references to Conversions
    std::vector<reco::ConversionRef>  conversions_;

    float hadOverEm_;
    bool pixelSeed_;
    float r9_;
    float r19_;
    float e5x5_;

  };
  
}

#endif
