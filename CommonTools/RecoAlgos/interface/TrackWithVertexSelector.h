#ifndef CommonTools_RecoAlgos_TrackWithVertexSelector_H
#define CommonTools_RecoAlgos_TrackWithVertexSelector_H

// Original Author:  Giovanni Petrucciani
//         Created:  Fri May 25 10:06:02 CEST 2007
// $Id: TrackWithVertexSelector.h,v 1.4 2010/04/07 08:56:18 gpetrucc Exp $

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TrackWithVertexSelector {
public:
  explicit TrackWithVertexSelector(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&iC)
      : TrackWithVertexSelector(iConfig, iC) {}
  explicit TrackWithVertexSelector(const edm::ParameterSet &iConfig, edm::ConsumesCollector &iC);
  ~TrackWithVertexSelector();

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

  void init(const edm::Event &event, const edm::EventSetup &) { init(event); }
  void init(const edm::Event &event);

  bool operator()(const reco::Track &t) const;
  bool operator()(const reco::Track &t, const edm::Event &iEvent) {
    init(iEvent);
    return (*this)(t);
  }

  bool operator()(const reco::TrackRef &t) const;
  bool operator()(const reco::TrackRef &t, const edm::Event &iEvent) {
    init(iEvent);
    return (*this)(t);
  }
  bool testTrack(const reco::Track &t) const;
  bool testVertices(const reco::Track &t, const reco::VertexCollection &vtxs) const;

  bool testTrack(const reco::TrackRef &t) const;
  bool testVertices(const reco::TrackRef &t, const reco::VertexCollection &vtxs) const;

private:
  const uint32_t numberOfValidHits_;
  const uint32_t numberOfValidPixelHits_;
  const uint32_t numberOfLostHits_;
  const double normalizedChi2_;
  const double ptMin_, ptMax_, etaMin_, etaMax_;
  const double dzMax_, d0Max_;
  const double ptErrorCut_;
  const std::string quality_;

  const uint32_t nVertices_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::EDGetTokenT<edm::ValueMap<float> > timesToken_, timeResosToken_;
  const bool vtxFallback_;
  const double zetaVtx_, rhoVtx_, nSigmaDtVertex_;

  reco::VertexCollection const *vcoll_ = nullptr;
  edm::ValueMap<float> const *timescoll_ = nullptr;
  edm::ValueMap<float> const *timeresoscoll_ = nullptr;
  typedef math::XYZPoint Point;
};

#endif
