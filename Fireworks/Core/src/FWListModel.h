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
// $Id: FWListModel.h,v 1.3 2008/03/05 19:57:37 chrjones Exp $
//

// system include files
#include "TEveElement.h"
#include "TNamed.h"
// user include files
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/src/FWListItemBase.h"

// forward declarations
class TObject;

class FWListModel : public TEveElement, public TNamed, public FWListItemBase
{

   public:
      FWListModel(const FWModelId& iId = FWModelId() );
      virtual ~FWListModel();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void SetMainColor(Color_t);
      virtual void SetRnrState(Bool_t rnr);
      ClassDef(FWListModel,0);

      virtual Bool_t CanEditMainColor() const;
      virtual Bool_t SingleRnrState() const;

      virtual bool doSelection(bool iToggleSelection);

      void openDetailView() const;
   private:
      FWListModel(const FWListModel&); // stop default

      const FWListModel& operator=(const FWListModel&); // stop default

      // ---------- member data --------------------------------
      FWModelId m_id;
      Color_t m_color;
};


#endif
