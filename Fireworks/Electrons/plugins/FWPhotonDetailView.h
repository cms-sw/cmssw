// -*- C++ -*-
#ifndef Fireworks_Electrons_FWPhotonDetailView_h
#define Fireworks_Electrons_FWPhotonDetailView_h
//
// Package:     Calo
// Class  :     FWPhotonDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"

class FWECALDetailViewBuilder;
class TEveCaloData;
namespace reco {
  class Photon;
}

class FWPhotonDetailView : public FWDetailViewGL<reco::Photon> {
public:
  FWPhotonDetailView();
  ~FWPhotonDetailView() override;

  using FWDetailViewGL<reco::Photon>::build;
  void build(const FWModelId& id, const reco::Photon*) override;
  using FWDetailViewGL<reco::Photon>::setTextInfo;
  void setTextInfo(const FWModelId& id, const reco::Photon*) override;

  FWPhotonDetailView(const FWPhotonDetailView&) = delete;                   // stop default
  const FWPhotonDetailView& operator=(const FWPhotonDetailView&) = delete;  // stop default

private:
  void addSceneInfo(const reco::Photon*, TEveElementList*);

  TEveCaloData* m_data;
  FWECALDetailViewBuilder* m_builder;
};

#endif
