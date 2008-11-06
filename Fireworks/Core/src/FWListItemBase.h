#ifndef Fireworks_Core_FWListItemBase_h
#define Fireworks_Core_FWListItemBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListItemBase
//
/**\class FWListItemBase FWListItemBase.h Fireworks/Core/interface/FWListItemBase.h

 Description: Base class for items to be shown in the list tree

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 11 14:35:33 CDT 2008
// $Id: FWListItemBase.h,v 1.1 2008/03/11 23:15:55 chrjones Exp $
//

// system include files
#include "Rtypes.h"

// user include files

// forward declarations

class FWListItemBase
{

   public:
   FWListItemBase() {}
   virtual ~FWListItemBase();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      //returns true if this object should be passed directly to the editor
      virtual bool doSelection(bool toggleSelection) = 0;

   ClassDef(FWListItemBase,0);

   private:
      FWListItemBase(const FWListItemBase&); // stop default

      const FWListItemBase& operator=(const FWListItemBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
