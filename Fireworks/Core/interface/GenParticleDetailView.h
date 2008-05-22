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
// $Id: GenParticleDetailView.h,v 1.1 2008/03/20 04:00:27 jmuelmen Exp $
//

// system include files

// user include files
#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;

class GenParticleDetailView : public FWDetailView {
     
public:
     GenParticleDetailView();
     virtual ~GenParticleDetailView();
     
     virtual void build (TEveElementList **product, const FWModelId &id);

protected:
     void setItem (const FWEventItem *iItem) { m_item = iItem; }
     void getCenter( Double_t* vars )
     {
	vars[0] = rotation_center[0];
	vars[1] = rotation_center[1];
	vars[2] = rotation_center[2];
     }
   
private:
     GenParticleDetailView(const GenParticleDetailView&); // stop default
     const GenParticleDetailView& operator=(const GenParticleDetailView&); // stop default
     
     // ---------- member data --------------------------------
     const FWEventItem* m_item;
     void resetCenter() { 
	rotation_center[0] = 0;
	rotation_center[1] = 0;
	rotation_center[2] = 0;
     }
	
};

#endif
