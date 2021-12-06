// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackHitsDetailView

#include "Rtypes.h"
#include "TVector3.h"
#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class TGLEmbeddedViewer;
class FWIntValueListener;
class TGSlider;
namespace reco {
  class Track;
}

class FWTrackHitsDetailView : public FWDetailViewGL<reco::Track>, public CSGActionSupervisor {
public:
  FWTrackHitsDetailView();
  ~FWTrackHitsDetailView() override;

  void pickCameraCenter();
  void transparencyChanged(int);
  void rnrLabels();

  void setBackgroundColor(Color_t) override;

protected:
  TEveElementList* m_modules;
  TEveElementList* m_moduleLabels;
  TEveElementList* m_hits;
  TGSlider* m_slider;
  FWIntValueListener* m_sliderListener;

public:
  FWTrackHitsDetailView(const FWTrackHitsDetailView&) = delete;                   // stop default
  const FWTrackHitsDetailView& operator=(const FWTrackHitsDetailView&) = delete;  // stop default

private:
  using FWDetailView<reco::Track>::build;
  void build(const FWModelId& id, const reco::Track*) override;
  using FWDetailView<reco::Track>::setTextInfo;
  void setTextInfo(const FWModelId& id, const reco::Track*) override;
  void makeLegend(void);

  void addTrackerHits3D(std::vector<TVector3>& points, class TEveElementList* tList, Color_t color, int size);

  void addHits(const reco::Track& track, const FWEventItem* iItem, TEveElement* trkList, bool addNearbyHits);
  void addModules(const reco::Track& track, const FWEventItem* iItem, TEveElement* trkList, bool addLostHits);

  TLegend* m_legend;
};
