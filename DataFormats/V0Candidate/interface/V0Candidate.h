#ifndef CANDIDATE_V0CANDIDATE_H
#define CANDIDATE_V0CANDIDATE_H
//#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace reco{

  class V0Candidate {


  public:
    V0Candidate();
    V0Candidate( const VertexCompositeCandidate& theCandIn, 
		 const reco::Track& thePosTk, 
		 const reco::Track& theNegTk );
    /*V0Candidate( const VertexCompositeCandidate& theCandIn,
		 const reco::TrackRef thePosTkRef,
		 const reco::TrackRef theNegTkRef );*/

    inline void setPosTrack( const reco::Track& thePosTk ) { posDaughter = thePosTk; }
    //inline void setPosTrack( const reco::TrackRef& thePosTkRef ) { posDaughter = *thePosTkRef; }
    inline void setNegTrack( const reco::Track& theNegTk ) { negDaughter = theNegTk; }
    //inline void setNegTrack( const reco::TrackRef& theNegTkRef ) { negDaughter = *theNegTkRef; }
    inline void setCand( const reco::VertexCompositeCandidate& theVee ) {
      theCand = theVee;
      isGoodV0 = true;
    }
    inline void setPosMomentum( const GlobalVector& posP ) { posMomentum = posP; }
    inline void setNegMomentum( const GlobalVector& negP ) { negMomentum = negP; }
    inline void setPosNPixelHits( const int posNPix ) { posNPixelHits = posNPix; }
    inline void setNegNPixelHits( const int negNPix ) { negNPixelHits = negNPix; }
    inline void setPosNStripHits( const int posNStr ) { posNStripHits = posNStr; }
    inline void setNegNStripHits( const int negNStr ) { negNStripHits = negNStr; }

    inline reco::Track getPosTrack() const { return posDaughter; }
    inline reco::Track getNegTrack() const { return negDaughter; }
    inline GlobalVector getPosTkP() const { return posMomentum; }
    inline GlobalVector getNegTkP() const { return negMomentum; }
    inline int getPosNPixelHits() const { return posNPixelHits; }
    inline int getPosNStripHits() const { return posNStripHits; }
    inline int getNegNPixelHits() const { return negNPixelHits; }
    inline int getNegNStripHits() const { return negNStripHits; }
    inline reco::VertexCompositeCandidate getVtxCC() const { return theCand; }
    inline bool isValid() { return isGoodV0; }

    //void setVertex( const Vertex & vtxIn );
 private:
    reco::VertexCompositeCandidate theCand;
    reco::Track posDaughter;
    reco::Track negDaughter;
    GlobalVector posMomentum;
    GlobalVector negMomentum;
    int posNPixelHits;
    int negNPixelHits;
    int posNStripHits;
    int negNStripHits;

    bool isGoodV0;
    
  };

}

#endif
