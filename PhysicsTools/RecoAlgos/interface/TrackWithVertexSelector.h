#ifndef PhysicsTools_RecoAlgos_TrackWithVertexSelector_H
#define PhysicsTools_RecoAlgos_TrackWithVertexSelector_H

// Original Author:  Giovanni Petrucciani
//         Created:  Fri May 25 10:06:02 CEST 2007
// $Id: TrackWithVertexSelector.h,v 1.2 2007/09/11 21:41:23 ratnik Exp $


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class TrackWithVertexSelector {
   public:
      explicit TrackWithVertexSelector(const edm::ParameterSet&);
      ~TrackWithVertexSelector();
      bool operator()(const reco::Track &t, const edm::Event &iEvent) const ;
      bool operator()(const reco::Track &t, const reco::VertexCollection &vtxs) const;
      bool testTrack(const reco::Track &t) const ;
      bool testVertices(const reco::Track &t, const reco::VertexCollection &vtxs) const ; 
   private:
      uint32_t numberOfValidHits_;
      uint32_t numberOfValidPixelHits_;
      uint32_t numberOfLostHits_;
      double   normalizedChi2_;
      double   ptMin_, ptMax_, etaMin_, etaMax_;
      double   dzMax_,   d0Max_;

      uint32_t      nVertices_;
      edm::InputTag vertexTag_;
      bool          vtxFallback_;
      double        zetaVtx_, rhoVtx_;

      typedef math::XYZPoint Point;
      bool testPoint(const Point &point, const Point &vtx) const ;
};

#endif
