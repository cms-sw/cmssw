// -*- C++ -*-
#ifndef Fireworks_Electrons_FWElectronDetailView_h
#define Fireworks_Electrons_FWElectronDetailView_h

//
// Package:     Electrons
// Class  :     FWElectronDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

class TEveCaloDataVec;
class TEveCaloLego;
class TLegend;
class FWECALDetailViewBuilder;
class TEveCaloData;
class TGSlider;

namespace reco {
  class GsfElectron;
}

class FWElectronDetailView : public FWDetailViewGL<reco::GsfElectron> {
public:
  FWElectronDetailView();
  ~FWElectronDetailView() override;

private:
  FWElectronDetailView(const FWElectronDetailView &) = delete;                   // stop default
  const FWElectronDetailView &operator=(const FWElectronDetailView &) = delete;  // stop default

  using FWDetailViewGL<reco::GsfElectron>::build;
  void build(const FWModelId &id, const reco::GsfElectron *) override;

  using FWDetailViewGL<reco::GsfElectron>::setTextInfo;
  void setTextInfo(const FWModelId &id, const reco::GsfElectron *) override;

  double deltaEtaSuperClusterTrackAtVtx(const reco::GsfElectron &);
  double deltaPhiSuperClusterTrackAtVtx(const reco::GsfElectron &);
  void addTrackPointsInCaloData(const reco::GsfElectron *, TEveCaloLego *);

  void addSceneInfo(const reco::GsfElectron *i, TEveElementList *tList);
  void drawCrossHair(const reco::GsfElectron *, TEveCaloLego *, TEveElementList *);

  Bool_t checkRange(Double_t &, Double_t &, Double_t &, Double_t &, Double_t, Double_t);

  TEveCaloData *m_data;
  FWECALDetailViewBuilder *m_builder;
  TLegend *m_legend;
};

#endif
