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
// $Id: FWPhotonDetailView.h,v 1.1 2009/03/31 23:27:20 jmuelmen Exp $
//

// user include files
#include "FWECALDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class FWPhotonDetailView : public FWECALDetailView<reco::Photon> {

public:
     FWPhotonDetailView();
     virtual ~FWPhotonDetailView();
     
     virtual TEveElement* build (const FWModelId &id, const reco::Photon*);
     
protected:
     virtual bool	drawTrack () { return false; }
     TEveElement* build_projected (const FWModelId &id, const reco::Photon*);
     virtual class TEveElementList *makeLabels (const reco::Photon &);
     
private:
     FWPhotonDetailView(const FWPhotonDetailView&); // stop default
     const FWPhotonDetailView& operator=(const FWPhotonDetailView&); // stop default
};

#endif
