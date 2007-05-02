#ifndef EgammaCandidates_ConvertedPhoton_h
#define EgammaCandidates_ConvertedPhoton_h
/** \class reco::ConvertedPhoton ConvertedPhoton.h DataFormats/EgammaCandidates/interface/ConvertedPhoton.h
 *
 * Reco Candidates with an ConvertedPhoton component
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: ConvertedPhoton.h,v 1.4 2007/01/26 16:24:39 nancy Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhotonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


namespace reco {
    class ConvertedPhoton : public RecoCandidate {
  public:
    /// default constructor
    ConvertedPhoton() : RecoCandidate() { }

  
    ConvertedPhoton( const reco::SuperClusterRef sc, const std::vector<reco::TrackRef> tr, Charge q, const LorentzVector & p4, const Point & vtx, const Point & convVtx  );
    /// destructor
    virtual ~ConvertedPhoton();
    /// returns a clone of the candidate
    ConvertedPhoton * clone() const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const ;

    // vector of references to  tracks
    std::vector<reco::TrackRef> tracks() const ; 
    // Bool flagging objects having track size >0
    bool isConverted() const;
    // Number of tracks= 0,1,2
    unsigned int nTracks() const {return  tracks().size(); }
    // if nTracks=2 
    double pairInvariantMass() const {return invMass_;}
    // Delta cotg(Theta) where Theta is the angle in the (y,z) plane between the two tracks 
    double pairCotThetaSeparation() const {return dCotTheta_;}
    // Conversion tracks momentum 
    GlobalVector  pairMomentum() const {return momTracks_;}
    // Phi  
    double pairMomentumPhi() const {return  phiTracks_;}
    // Eta 
    double pairMomentumEta() const {return etaTracks_;}
    // Pt from tracks divided by the super cluster transverse energy
    double pairPtOverEtSC() const {return ptOverEtSC_;}
    // Super Cluster energy divided by tracks momentum
    double EoverP() const {return ep_;}
    // returns the position of the conversion vertex
    const Point & convVertexPosition() const { return theConversionVertex_ ; }
   /// reference to one of multiple Tracks: implements the method inherited from RecoCandidate
     reco::TrackRef track( size_t ) const;


  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
   

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    std::vector<reco::TrackRef>  tracks_;
    reco::Particle::Point theConversionVertex_;

    void makePairInvariantMass() ;
    void makePairCotThetaSeparation();
    void makePairMomentum() ;
    void makePairMomentumEta() ;
    void makePairMomentumPhi() ;
    void makePairPtOverEtSC() ;
    void makeEoverP() ;

    double invMass_;
    double dCotTheta_;
    double etaTracks_;
    double phiTracks_;
    GlobalVector  momTracks_;
    double ptOverEtSC_;
    double ep_;


  };
  
}

#endif
