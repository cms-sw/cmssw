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
// $Id$
//

// system include files
#include "TEveCaloData.h"

// user include files
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"

// forward declarations
class TH2F;
class FWEventItem;

class FWFromSliceSelector {
public:
   FWFromSliceSelector( TH2F* iHist,
                       const FWEventItem*);
   void doSelect(const TEveCaloData::CellId_t&);
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
   
   void addSliceSelector(int iSlice, const FWFromSliceSelector&);
private:
   FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector&); // stop default
   
   const FWFromTEveCaloDataSelector& operator=(const FWFromTEveCaloDataSelector&); // stop default
   
   // ---------- member data --------------------------------
   std::vector<FWFromSliceSelector> m_sliceSelectors;
   TEveCaloData* m_data;
   

};


#endif
