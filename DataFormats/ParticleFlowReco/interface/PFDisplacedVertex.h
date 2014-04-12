#ifndef DataFormat_ParticleFlowReco_PFDisplacedVertex_h
#define DataFormat_ParticleFlowReco_PFDisplacedVertex_h 

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>
#include <string>
#include <iostream>

namespace reco {

  
  /// \brief Block of elements
  /*!
    \author Maxime Gouzevitch
    \date November 2009

    A DisplacedVertex is an extension of Vector with some additionnal informations
    tracks's hit-vertex distances, tracks types and the expected vertex type.
  */
  
  class PFDisplacedVertex : public Vertex{

  public:

    /// Information on the distance between track's hits and the Vertex
    typedef std::pair <unsigned int, unsigned int> PFTrackHitInfo;
    typedef std::pair <PFTrackHitInfo, PFTrackHitInfo> PFTrackHitFullInfo;

    /// Mass hypothesis enum
    enum M_Hypo {
      M_CUSTOM,
      M_MASSLESS,
      M_PION,
      M_KAON,
      M_LAMBDA
    };


    /// Classification of tracks according to the position with respect 
    /// to the Vertex. A Merged track is a track which has at least 
    /// two hits before and two hits after the vertex. It may come from 
    /// a primary track merged with a low quality secondary track.
    enum VertexTrackType {
      T_NOT_FROM_VERTEX,
      T_TO_VERTEX,
      T_FROM_VERTEX,
      T_MERGED
    };

    /// Classification of vertex according to different parameters such as the 
    /// Number of tracks, the invariant mass etc...
    enum VertexType {
      ANY = 0,
      FAKE = 1,
      LOOPER = 2,
      NUCL = 10,
      NUCL_LOOSE = 11,
      NUCL_KINK = 12,
      CONVERSION = 20,
      CONVERSION_LOOSE = 21,
      CONVERTED_BREMM = 22,
      K0_DECAY = 30,
      LAMBDA_DECAY = 31,
      LAMBDABAR_DECAY = 32,
      KPLUS_DECAY = 40,
      KMINUS_DECAY = 41,
      KPLUS_DECAY_LOOSE = 42,
      KMINUS_DECAY_LOOSE = 43,
      BSM_VERTEX = 100
    };


    /// Default constructor
    PFDisplacedVertex();

    /// Constructor from the reco::Vertex
    PFDisplacedVertex(reco::Vertex&);

    /// Add a new track to the vertex
    void addElement( const TrackBaseRef & r, const Track & refTrack, 
		     const PFTrackHitFullInfo& hitInfo , 
		     VertexTrackType trackType = T_NOT_FROM_VERTEX, float w=1.0 );

    /// Clean the tracks collection and all the associated collections
    void cleanTracks();
    
    /// Set the type of this vertex
    void setVertexType(VertexType vertexType) {vertexType_ = vertexType;}

    /// Estimate the direction of the vertex. This function produced a unitary vector.
    /// It is calculated the axis linking the primary vertex of the event pvtx to this vertex
    void setPrimaryDirection(const math::XYZPoint& pvtx);

    /// Get the type of this vertex
    VertexType vertexType(){return vertexType_;}

    const std::vector < PFTrackHitFullInfo > trackHitFullInfos() const
      {return trackHitFullInfos_;}

    const std::vector <VertexTrackType> trackTypes() const
      {return trackTypes_;}


    /// -------- Provide useful information -------- ///

    /// If a primary track was identified
    const bool isTherePrimaryTracks() const 
    {return isThereKindTracks(T_TO_VERTEX);}

    /// If a merged track was identified
    const bool isThereMergedTracks() const
    {return isThereKindTracks(T_MERGED);}

    /// If a secondary track was identified
    const bool isThereSecondaryTracks() const
    {return isThereKindTracks(T_FROM_VERTEX);}

    /// If there is a track which was not identified
    const bool isThereNotFromVertexTracks() const
    {return isThereKindTracks(T_NOT_FROM_VERTEX);}




    /// Is a primary track was identified
    const bool isPrimaryTrack(const reco::TrackBaseRef& originalTrack) const 
    {
      size_t itrk = trackPosition(originalTrack);
      return isTrack(itrk, T_TO_VERTEX);
    }

    /// Is a secondary track was identified
    const bool isSecondaryTrack(const reco::TrackBaseRef& originalTrack) const 
    {
      size_t itrk = trackPosition(originalTrack);
      return isTrack(itrk, T_FROM_VERTEX);
    }

    /// Is a secondary track was identified
    const bool isMergedTrack(const reco::TrackBaseRef& originalTrack) const 
    {
      size_t itrk = trackPosition(originalTrack);
      return isTrack(itrk, T_MERGED);
    }

    const PFTrackHitFullInfo trackHitFullInfo(const reco::TrackBaseRef& originalTrack) const{
      size_t itrk = trackPosition(originalTrack);
      return trackHitFullInfos_[itrk];
    }



    /// Is primary or merged track
    const bool isIncomingTrack(const reco::TrackBaseRef& originalTrack) const 
    {
      size_t itrk = trackPosition(originalTrack);
      return isTrack(itrk, T_MERGED) || isTrack(itrk, T_TO_VERTEX);
    }
 
    /// Is secondary track
    const bool isOutgoingTrack(const reco::TrackBaseRef& originalTrack) const 
    {
      size_t itrk = trackPosition(originalTrack);
      return isTrack(itrk, T_FROM_VERTEX);    
    }


    /// Number of primary tracks was identified
    const int nPrimaryTracks() const
    {return nKindTracks(T_TO_VERTEX);}

    /// Number of merged tracks was identified
    const int nMergedTracks() const
    {return nKindTracks(T_MERGED);}

    /// Number of secondary tracks was identified
    const int nSecondaryTracks() const
    {return nKindTracks(T_FROM_VERTEX);}

    /// Number of tracks which was not identified
    const int nNotFromVertexTracks() const
    {return nKindTracks(T_NOT_FROM_VERTEX);}

    /// Number of tracks
    const int nTracks() const {return trackTypes_.size();}

    //    const reco::VertexTrackType vertexTrackType(reco::TrackBaseRef tkRef) const;

    /// Momentum of secondary tracks calculated with a mass hypothesis. Some of those
    /// hypothesis are default: "PI" , "KAON", "LAMBDA", "MASSLESS", "CUSTOM"
    /// the value of custom shall be then provided in mass variable
    const math::XYZTLorentzVector 
      secondaryMomentum(std::string massHypo = "PI", 
			bool useRefitted = true, double mass = 0.0) const 
      {return momentum(massHypo, T_FROM_VERTEX, useRefitted, mass);}

    /// Momentum of primary or merged track calculated with a mass hypothesis.
    const math::XYZTLorentzVector 
      primaryMomentum(std::string massHypo = "PI", 
		      bool useRefitted = true, double mass = 0.0) const
      {return momentum(massHypo, T_TO_VERTEX, useRefitted, mass);}



    /// Momentum of secondary tracks calculated with a mass hypothesis. Some of those
    /// hypothesis are default: "PI" , "KAON", "LAMBDA", "MASSLESS", "CUSTOM"
    /// the value of custom shall be then provided in mass variable
    const math::XYZTLorentzVector 
      secondaryMomentum(M_Hypo massHypo, 
			bool useRefitted = true, double mass = 0.0) const 
      {return momentum(massHypo, T_FROM_VERTEX, useRefitted, mass);}

    /// Momentum of primary or merged track calculated with a mass hypothesis.
    const math::XYZTLorentzVector 
      primaryMomentum(M_Hypo massHypo, 
		      bool useRefitted = true, double mass = 0.0) const
      {return momentum(massHypo, T_TO_VERTEX, useRefitted, mass);}


    void calcKinematics(){
      defaultPrimaryMomentum_ = momentum("PI", T_TO_VERTEX, false, 0.0);
      defaultSecondaryMomentum_ = momentum("PI", T_FROM_VERTEX, true, 0.0);
    }


    const double secondaryPt() const 
      {return defaultPrimaryMomentum_.Pt();}

    /// Momentum of primary or merged track calculated with a mass hypothesis.
    const double primaryPt() const
      {return defaultSecondaryMomentum_.Pt();}



    /// Total Charge
    const int totalCharge() const;

    /// Angle PrimaryVertex-DisplacedVertex (or primary track if there is)
    /// And secondary momentum
    const double angle_io() const;


    /// Primary Direction
    const math::XYZVector primaryDirection() const;



    bool isFake() const { return vertexType_ == FAKE;}
    bool isLooper() const { return vertexType_ ==  LOOPER;}
    bool isNucl() const { return vertexType_ ==  NUCL;}
    bool isNucl_Loose() const { return vertexType_ ==  NUCL_LOOSE;}
    bool isNucl_Kink() const { return vertexType_ ==   NUCL_KINK;}
    bool isConv() const { return vertexType_ ==   CONVERSION;}
    bool isConv_Loose() const { return vertexType_ ==   CONVERSION_LOOSE;}
    bool isConvertedBremm() const { return vertexType_ ==   CONVERTED_BREMM;}
    bool isK0() const { return vertexType_ ==   K0_DECAY;}
    bool isLambda() const { return vertexType_ ==   LAMBDA_DECAY;}
    bool isLambdaBar() const { return vertexType_ ==   LAMBDABAR_DECAY;}
    bool isKplus() const { return vertexType_ ==   KPLUS_DECAY;}
    bool isKminus() const { return vertexType_ ==   KMINUS_DECAY;}
    bool isKplus_Loose() const { return vertexType_ ==   KPLUS_DECAY_LOOSE;}
    bool isKminus_Loose() const { return vertexType_ ==   KMINUS_DECAY_LOOSE;}
    bool isBSM() const { return vertexType_ ==   BSM_VERTEX;}


    std::string nameVertexType() const;

    /// cout function
    void Dump(std::ostream& out = std::cout) const;

  private:

    /// --------- TOOLS -------------- ///

    /// Common tool used to know if there are tracks of a given Kind
    const bool isThereKindTracks(VertexTrackType) const;

    /// Common tool used to get the number of tracks of a given Kind
    const int nKindTracks(VertexTrackType) const;

    /// Common tool to calculate the momentum vector of tracks with a given Kind
    const  math::XYZTLorentzVector momentum(std::string, 
					    VertexTrackType,
					    bool, double mass) const;

    /// Common tool to calculate the momentum vector of tracks with a given Kind
    const  math::XYZTLorentzVector momentum(M_Hypo massHypo, 
					    VertexTrackType,
					    bool, double mass) const;

    /// Get the mass with a given hypothesis
    const double getMass2(M_Hypo, double) const;

    const size_t trackPosition(const reco::TrackBaseRef& originalTrack) const;

    const bool isTrack(size_t itrk, VertexTrackType T) const {
      return  trackTypes_[itrk] == T;
    }


    /// -------- MEMBERS -------- ///

    /// This vertex type
    VertexType vertexType_;

    /// Types of the tracks associated to the vertex 
    std::vector < VertexTrackType > trackTypes_;

    /// Information on the distance between track's hits and the Vertex
    std::vector < PFTrackHitFullInfo > trackHitFullInfos_;

    math::XYZVector primaryDirection_;

    math::XYZTLorentzVector defaultPrimaryMomentum_;
    math::XYZTLorentzVector defaultSecondaryMomentum_;

    
  };
}

#endif


  
