#ifndef Fireworks_Calo_ElectronDetailView_h
#define Fireworks_Calo_ElectronDetailView_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronDetailView
// 
/**\class ElectronDetailView ElectronDetailView.h Fireworks/Calo/interface/ElectronDetailView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ElectronDetailView.h,v 1.3 2008/03/06 22:48:31 jmuelmen Exp $
//

// system include files

#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;

class ElectronDetailView : public FWDetailView {
     
public:
     ElectronDetailView();
     virtual ~ElectronDetailView();
     
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
     ElectronDetailView(const ElectronDetailView&); // stop default
     const ElectronDetailView& operator=(const ElectronDetailView&); // stop default
     
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
