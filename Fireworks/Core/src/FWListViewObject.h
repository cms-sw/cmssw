#ifndef Fireworks_Core_FWListViewObject_h
#define Fireworks_Core_FWListViewObject_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListViewObject
// 
/**\class FWListViewObject FWListViewObject.h Fireworks/Core/interface/FWListViewObject.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 09:02:41 CDT 2008
// $Id$
//

// system include files
#include "TEveElement.h"
#include "TNamed.h"

// user include files

// forward declarations
class FWViewBase;

class FWListViewObject : public TEveElement, public TNamed
{

   public:
      FWListViewObject(const char* iName, FWViewBase* iView);
      //virtual ~FWListViewObject();

      // ---------- const member functions ---------------------
      virtual Bool_t CanEditMainColor() const;
      virtual Bool_t CanEditElement() const { return kFALSE; }
      FWViewBase* view() const { return m_view; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      ClassDef(FWListViewObject,0);

   private:
      FWListViewObject(const FWListViewObject&); // stop default

      const FWListViewObject& operator=(const FWListViewObject&); // stop default

      // ---------- member data --------------------------------
      FWViewBase* m_view;
};


#endif
