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
// $Id: FWElectronDetailView.h,v 1.3 2009/03/23 15:54:12 amraktad Exp $
//


// user include files
#include "Fireworks/Core/interface/FWDetailView.h"


#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


class FWElectronDetailView : public FWDetailView<reco::GsfElectron> {

public:
   FWElectronDetailView();
   virtual ~FWElectronDetailView();

   virtual TEveElement* build (const FWModelId &id, const reco::GsfElectron*);

protected:
   void setItem (const FWEventItem *iItem) {
      m_item = iItem;
   }
   TEveElement* build_projected (const FWModelId &id, const reco::GsfElectron*);

   TEveElementList *makeLabels (const reco::GsfElectron &);
   void getEcalCrystalsEndcap (std::vector<DetId> *, 
			       int ix, int iy, int iz);
   void getEcalCrystalsBarrel (std::vector<DetId> *, 
			       int ieta, int iphi);
   void fillData (const std::vector<DetId> &detids,
                  TEveCaloDataVec *data, 
                  double phi_seed);
   void fillReducedData (const std::vector<DetId> &detids,
			 TEveCaloDataVec *data);

private:
   FWElectronDetailView(const FWElectronDetailView&); // stop default
   const FWElectronDetailView& operator=(const FWElectronDetailView&); // stop default

   // ---------- member data --------------------------------
   void resetCenter() {
      rotationCenter()[0] = 0;
      rotationCenter()[1] = 0;
      rotationCenter()[2] = 0;
   }

   void getCenter( Double_t* vars )
   {
      vars[0] = rotationCenter()[0];
      vars[1] = rotationCenter()[1];
      vars[2] = rotationCenter()[2];
   }

   const FWEventItem* m_item;

   bool  m_coordEtaPhi; // use XY coordinate if EndCap, else EtaPhi

   const EcalRecHitCollection *m_barrel_hits;
   const EcalRecHitCollection *m_endcap_hits;
   const EcalRecHitCollection *m_endcap_reduced_hits;
   std::vector<DetId> seed_detids;

};

#endif
