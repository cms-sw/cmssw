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
// $Id: FWElectronDetailView.h,v 1.2 2009/03/18 12:45:36 amraktad Exp $
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
   TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
                                           const std::vector<class DetId> &);
   TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
                                           double eta, double phi,
                                           int n_eta = 5, int n_phi = 10);
   TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
                                           const std::vector<class DetId> &);
   TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
                                           double x, double y, int iz,
                                           int n_x = 5, int n_y = 5);

   void fillData (const std::vector<DetId> &detids,
                  TEveCaloDataVec *data, 
                  double phi_seed);

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
   std::vector<DetId> seed_detids;

};

#endif
