// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWConvTrackHitsDetailView

#ifndef Fireworks_Electrons_FWConversionDetailView_h
#define Fireworks_Electrons_FWConversionDetailView_h

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TVector3.h"

class TGLEmbeddedViewer;
class TGTextButton;

namespace reco {
  class Conversion;
}

class FWConvTrackHitsDetailView : public FWDetailViewGL<reco::Conversion>, public CSGActionSupervisor {
public:
  FWConvTrackHitsDetailView();
  ~FWConvTrackHitsDetailView() override;

  void pickCameraCenter();
  void rnrLabels();
  void rnrModules();
  void rnrHits();

  void camera1Callback();
  void camera2Callback();
  void camera3Callback();
  void switchProjection();

  FWConvTrackHitsDetailView(const FWConvTrackHitsDetailView&) = delete;                   // stop default
  const FWConvTrackHitsDetailView& operator=(const FWConvTrackHitsDetailView&) = delete;  // stop default

private:
  using FWDetailViewGL<reco::Conversion>::build;
  void build(const FWModelId& id, const reco::Conversion*) override;
  using FWDetailViewGL<reco::Conversion>::setTextInfo;
  void setTextInfo(const FWModelId& id, const reco::Conversion*) override;

  void addTrackerHits3D(std::vector<TVector3>& points, class TEveElementList* tList, Color_t color, int size);

  void addHits(const reco::Track& track, const FWEventItem* iItem, TEveElement* trkList, bool addNearbyHits);
  void addModules(const reco::Track& track, const FWEventItem* iItem, TEveElement* trkList, bool addLostHits);

  TEveElementList* m_modules;
  TEveElementList* m_moduleLabels;
  TEveElementList* m_hits;
  TEveElement* m_calo3D;

  TLegend* m_legend;
  bool m_orthographic;
  CSGAction* m_camTypeAction;
};

#endif
