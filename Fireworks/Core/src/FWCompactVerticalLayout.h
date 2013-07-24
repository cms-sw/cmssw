#ifndef Fireworks_Core_FWCompactVerticalLayout_h
#define Fireworks_Core_FWCompactVerticalLayout_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCompactVerticalLayout
// 
/**\class FWCompactVerticalLayout FWCompactVerticalLayout.h Fireworks/Core/interface/FWCompactVerticalLayout.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 17 12:10:41 CDT 2009
// $Id: FWCompactVerticalLayout.h,v 1.1 2009/03/18 15:36:41 chrjones Exp $
//

// system include files
#include "TGLayout.h"

// user include files

// forward declarations

class FWCompactVerticalLayout : public TGVerticalLayout {

public:
   FWCompactVerticalLayout( TGCompositeFrame* iMain);
   virtual ~FWCompactVerticalLayout();
   
   // ---------- const member functions ---------------------
   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   ClassDef(FWCompactVerticalLayout,0)  // Vertical layout manager

private:
   FWCompactVerticalLayout(const FWCompactVerticalLayout&); // stop default
   
   const FWCompactVerticalLayout& operator=(const FWCompactVerticalLayout&); // stop default
   
   // ---------- member data --------------------------------
   
};


#endif
