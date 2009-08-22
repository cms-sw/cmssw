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
// $Id: FWPhotonDetailView.h,v 1.2 2009/04/23 17:06:49 jmuelmen Exp $
//

// user include files
#include "FWECALDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class TEveWindowSlot;

class FWPhotonDetailView : public FWECALDetailView<reco::Photon> {

public:
   FWPhotonDetailView();
   virtual ~FWPhotonDetailView();

   virtual void build (const FWModelId &id, const reco::Photon*, TEveWindowSlot*);

protected:
   virtual bool	drawTrack () { return false; }
   virtual void makeLegend (const reco::Photon &, TCanvas*);

private:
   FWPhotonDetailView(const FWPhotonDetailView&); // stop default
   const FWPhotonDetailView& operator=(const FWPhotonDetailView&); // stop default
};

#endif
