#ifndef Fireworks_Calo_FWFromTEveCaloDataSelector_h
#define Fireworks_Calo_FWFromTEveCaloDataSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWFromTEveCaloDataSelector
// 
/**\class FWFromTEveCaloDataSelector FWFromTEveCaloDataSelector.h Fireworks/Calo/interface/FWFromTEveCaloDataSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Oct 23 14:44:32 CDT 2009
// $Id: FWFromTEveCaloDataSelector.h,v 1.7 2010/06/02 18:49:07 amraktad Exp $
//

// system include files
#include "TEveCaloData.h"

// user include files
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"
#include "Fireworks/Calo/src/FWFromSliceSelector.h"

// forward declarations
class FWEventItem;
class FWModelChangeManager;

//==============================================================================

class FWFromTEveCaloDataSelector : public FWFromEveSelectorBase
{

public:
   FWFromTEveCaloDataSelector(TEveCaloData*);
   virtual ~FWFromTEveCaloDataSelector();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void doSelect();
   void doUnselect();   

   void addSliceSelector(int iSlice, FWFromSliceSelector*);
   void resetSliceSelector(int iSlice);
private:
   FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector&); // stop default
   
   const FWFromTEveCaloDataSelector& operator=(const FWFromTEveCaloDataSelector&); // stop default
   
   // ---------- member data --------------------------------
   std::vector<FWFromSliceSelector*> m_sliceSelectors;
   TEveCaloData* m_data; // cached
   FWModelChangeManager* m_changeManager;
  
};


#endif
