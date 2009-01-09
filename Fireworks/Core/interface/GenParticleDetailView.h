#ifndef Fireworks_Calo_GenParticleDetailView_h
#define Fireworks_Calo_GenParticleDetailView_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     GenParticleDetailView
//
/**\class GenParticleDetailView GenParticleDetailView.h Fireworks/Calo/interface/GenParticleDetailView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: GenParticleDetailView.h,v 1.2 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// user include files
#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;

class GenParticleDetailView : public FWDetailView<reco::GenParticle> {

public:
     GenParticleDetailView();
     virtual ~GenParticleDetailView();

     virtual TEveElement* build (const FWModelId &id, const reco::GenParticle*);

protected:
     void setItem (const FWEventItem *iItem) { m_item = iItem; }

private:
     GenParticleDetailView(const GenParticleDetailView&); // stop default
     const GenParticleDetailView& operator=(const GenParticleDetailView&); // stop default

     // ---------- member data --------------------------------
     const FWEventItem* m_item;
};

#endif
