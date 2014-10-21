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
namespace reco {
class Photon;
}

class FWPhotonDetailView : public FWDetailViewGL<reco::Photon> {

public:
   FWPhotonDetailView();
   virtual ~FWPhotonDetailView();

   using FWDetailViewGL<reco::Photon>::build;
   virtual void build (const FWModelId &id, const reco::Photon*);
   using FWDetailViewGL<reco::Photon>::setTextInfo;
   virtual void setTextInfo(const FWModelId &id, const reco::Photon*);

private:
   FWPhotonDetailView(const FWPhotonDetailView&); // stop default
   const FWPhotonDetailView& operator=(const FWPhotonDetailView&); // stop default

   void addSceneInfo(const reco::Photon*, TEveElementList*);

   TEveCaloData* m_data;
   FWECALDetailViewBuilder* m_builder;
};

#endif
