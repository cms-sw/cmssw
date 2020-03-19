// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWPFDetailView

#include "Rtypes.h"

#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
class TGLEmbeddedViewer;
class FWIntValueListener;
class TEveCaloLego;
class TGSlider;

namespace reco {
  //   class PFCandidate;
  class PFRecHit;
  class PFCluster;
  class PFRecTrack;
}  // namespace reco

class FWPFCandidateDetailView : public FWDetailViewGL<reco::PFCandidate>, public CSGActionSupervisor {
public:
  FWPFCandidateDetailView();
  ~FWPFCandidateDetailView() override;

protected:
private:
  FWPFCandidateDetailView(const FWPFCandidateDetailView &) = delete;                   // stop default
  const FWPFCandidateDetailView &operator=(const FWPFCandidateDetailView &) = delete;  // stop default

  using FWDetailView<reco::PFCandidate>::build;
  void build(const FWModelId &id, const reco::PFCandidate *) override;
  void setTextInfo(const FWModelId &id, const reco::PFCandidate *) override;
  void makeLegend(void);

  bool isPntInRng(float x, float y);

  void rangeChanged(int x);
  void plotEtChanged();
  void rnrHcalChanged();

  void buildGLEventScene();

  void voteMaxEtEVal(const std::vector<reco::PFRecHit> *hits);

  void addHits(const std::vector<reco::PFRecHit> *);
  void addClusters(const std::vector<reco::PFCluster> *);
  void addTracks(const std::vector<reco::PFRecTrack> *);

  float eta();
  float phi();

  float etaMin() { return eta() - m_range; }
  float etaMax() { return eta() + m_range; }
  float phiMin() { return phi() - m_range; }
  float phiMax() { return phi() + m_range; }

  float m_range;
  const reco::PFCandidate *m_candidate;

  TLegend *m_legend;

  TGSlider *m_slider;
  FWIntValueListener *m_sliderListener;

  TEveElementList *m_eventList;

  bool m_plotEt;

  bool m_rnrHcal;
};
