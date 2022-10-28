// -*- C++ -*-
#ifndef Fireworks_Electrons_FWMuonDetailView_h
#define Fireworks_Electrons_FWMuonDetailView_h

//
// Package:     Electrons
// Class  :     FWMuonDetailView
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"

class FWECALDetailViewBuilder;
class TEveCaloData;
namespace reco {
  class Muon;
}

class FWMuonDetailView : public FWDetailViewGL<reco::Muon> {
public:
  FWMuonDetailView();
  ~FWMuonDetailView() override;

  FWMuonDetailView(const FWMuonDetailView&) = delete;                   // stop default
  const FWMuonDetailView& operator=(const FWMuonDetailView&) = delete;  // stop default

private:
  void build(const FWModelId& id, const reco::Muon*) override;
  void setTextInfo(const FWModelId&, const reco::Muon*) override;

  void addSceneInfo(const reco::Muon* i, TEveElementList* tList);

  TEveCaloData* m_data;
  FWECALDetailViewBuilder* m_builder;
};

#endif
