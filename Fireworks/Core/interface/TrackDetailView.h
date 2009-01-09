#ifndef Fireworks_Calo_TrackDetailView_h
#define Fireworks_Calo_TrackDetailView_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     TrackDetailView
//
/**\class TrackDetailView TrackDetailView.h Fireworks/Calo/interface/TrackDetailView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: TrackDetailView.h,v 1.2 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;

class TrackDetailView : public FWDetailView<reco::Track> {

public:
     TrackDetailView();
     virtual ~TrackDetailView();

     virtual TEveElement* build (const FWModelId &id,const reco::Track*);

protected:
     void getCenter( Double_t* vars )
     {
	vars[0] = rotationCenter()[0];
	vars[1] = rotationCenter()[1];
	vars[2] = rotationCenter()[2];
     }

private:
     TrackDetailView(const TrackDetailView&); // stop default
     const TrackDetailView& operator=(const TrackDetailView&); // stop default

     // ---------- member data --------------------------------
     void resetCenter() {
	rotationCenter()[0] = 0;
	rotationCenter()[1] = 0;
	rotationCenter()[2] = 0;
     }

};

#endif
