#ifndef EgammaCandidates_Conversion_h
#define EgammaCandidates_Conversion_h
/** \class reco::Conversion Conversion.h DataFormats/EgammaCandidates/interface/Conversion.h
 *
 * 
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: Conversion.h,v 1.2 2007/12/10 19:05:24 nancy Exp $
 *
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h" 
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"



namespace reco {
    class Conversion  {
  public:

    // Default constructor
    Conversion() {}
   
    Conversion( const reco::SuperClusterRef sc, 
                     const std::vector<reco::TrackRef> tr, 
		     const std::vector<math::XYZPoint> trackPositionAtEcal , 
		     const reco::Vertex  &  convVtx,
		     const std::vector<reco::BasicClusterRef> & matchingBC);



    /// destructor
    virtual ~Conversion();
    /// returns a clone of the candidate
    Conversion * clone() const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const ;
    /// vector of references to  tracks
    std::vector<reco::TrackRef> tracks() const ; 
     /// returns  the reco conversion vertex
    const reco::Vertex & conversionVertex() const  { return theConversionVertex_ ; }
     /// positions of the track extrapolation at the ECAL front face
    const std::vector<math::XYZPoint> & ecalImpactPosition() const  {return thePositionAtEcal_;}
    //  pair of BC matching the tracks
    const std::vector<reco::BasicClusterRef>&  bcMatchingWithTracks() const { return theMatchingBCs_;}
    /// Bool flagging objects having track size >0
    bool isConverted() const;
    /// Number of tracks= 0,1,2
    unsigned int nTracks() const {return  tracks().size(); }
    /// if nTracks=2 returns the pair invariant mass
    double pairInvariantMass() const;
    /// Delta cot(Theta) where Theta is the angle in the (y,z) plane between the two tracks 
    double pairCotThetaSeparation() const;
    /// Conversion tracks momentum 
    GlobalVector  pairMomentum() const;
    /// Super Cluster energy divided by tracks momentum
    double EoverP() const;
    /// set primary event vertex used to define photon direction
    double zOfPrimaryVertexFromTracks() const;




  private:

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    /// reference to a vector Track references
    std::vector<reco::TrackRef>  tracks_;
    std::vector<math::XYZPoint>  thePositionAtEcal_;
    reco::Vertex theConversionVertex_;
    std::vector<reco::BasicClusterRef> theMatchingBCs_;

    /*    
    double makePairInvariantMass() const;
    double makePairCotThetaSeparation() const;
    GlobalVector makePairMomentum() const;
    double makePairPtOverEtSC() const;
    double makeEoverP() const;
    double makePrimaryVertexZ() const ;
    */


  };
  
}

#endif
