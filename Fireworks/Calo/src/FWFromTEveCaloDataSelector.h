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
// $Id: FWFromTEveCaloDataSelector.h,v 1.2 2009/10/28 14:39:59 chrjones Exp $
//

// system include files
#include "TEveCaloData.h"

// user include files
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"

// forward declarations
class TH2F;
class FWEventItem;
class FWModelChangeManager;

class FWFromSliceSelector {
public:
   FWFromSliceSelector( TH2F* iHist,
                       const FWEventItem*);
   void doSelect(const TEveCaloData::CellId_t&);
   void doUnselect(const TEveCaloData::CellId_t&);
   void clear();
   FWModelChangeManager* changeManager() const;
private:
    TH2F* m_hist;
   const FWEventItem* m_item;
};

class FWFromTEveCaloDataSelector : public FWFromEveSelectorBase
{

public:
   FWFromTEveCaloDataSelector(TEveCaloData*);
   //virtual ~FWFromTEveCaloDataSelector();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void doSelect();
   void doUnselect();
   
   void addSliceSelector(int iSlice, const FWFromSliceSelector&);
private:
   FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector&); // stop default
   
   const FWFromTEveCaloDataSelector& operator=(const FWFromTEveCaloDataSelector&); // stop default
   
   // ---------- member data --------------------------------
   std::vector<FWFromSliceSelector> m_sliceSelectors;
   TEveCaloData* m_data;
   FWModelChangeManager* m_changeManager;
   

};


#endif
