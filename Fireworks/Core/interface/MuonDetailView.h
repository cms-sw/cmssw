#ifndef Fireworks_Calo_MuonDetailView_h
#define Fireworks_Calo_MuonDetailView_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MuonDetailView
//
/**\class MuonDetailView MuonDetailView.h Fireworks/Calo/interface/MuonDetailView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MuonDetailView.h,v 1.1 2008/03/07 04:01:20 tdaniels Exp $
//

// system include files

#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;

class MuonDetailView : public FWDetailView {

public:
     MuonDetailView();
     virtual ~MuonDetailView();

     virtual void build (TEveElementList **product, const FWModelId &id);

protected:
     void setItem (const FWEventItem *iItem) { m_item = iItem; }
     void build (TEveElementList **product);
     void getCenter( Double_t* vars )
     {
	vars[0] = rotation_center[0];
	vars[1] = rotation_center[1];
	vars[2] = rotation_center[2];
     }

private:
     MuonDetailView(const MuonDetailView&); // stop default
     const MuonDetailView& operator=(const MuonDetailView&); // stop default

     // ---------- member data --------------------------------
     const FWEventItem* m_item;
     void resetCenter() {
	rotation_center[0] = 0;
	rotation_center[1] = 0;
	rotation_center[2] = 0;
     }

     Double_t rotation_center[3];
};


#endif
