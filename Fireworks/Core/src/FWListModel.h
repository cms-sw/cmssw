#ifndef Fireworks_Core_FWListModel_h
#define Fireworks_Core_FWListModel_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListModel
// 
/**\class FWListModel FWListModel.h Fireworks/Core/interface/FWListModel.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 17:20:20 EST 2008
// $Id$
//

// system include files
#include "TEveElement.h"
#include "TNamed.h"
// user include files

// forward declarations
class TObject;

class FWListModel : public TEveElement, public TNamed
{

   public:
      FWListModel(int iIndex=-1);
      virtual ~FWListModel();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void SetMainColor(Color_t);
      virtual void SetRnrSelf(Bool_t rnr);
      ClassDef(FWListModel,0);

      virtual Bool_t CanEditMainColor() const;
   
      void openDetailView() const;
   private:
      FWListModel(const FWListModel&); // stop default

      const FWListModel& operator=(const FWListModel&); // stop default

      // ---------- member data --------------------------------
      int m_index;
      Color_t m_color;
};


#endif
