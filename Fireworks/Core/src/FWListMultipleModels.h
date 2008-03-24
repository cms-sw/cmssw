#ifndef Fireworks_Core_FWListMultipleModels_h
#define Fireworks_Core_FWListMultipleModels_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListMultipleModels
// 
/**\class FWListMultipleModels FWListMultipleModels.h Fireworks/Core/interface/FWListMultipleModels.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 24 11:45:16 EDT 2008
// $Id$
//

// system include files
#include <set>
#include "TNamed.h"
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/src/FWListItemBase.h"

// forward declarations
class FWModelId;

class FWListMultipleModels : public TEveElement, public TNamed, public FWListItemBase
{

   public:
      FWListMultipleModels(const std::set<FWModelId>& iIds);
      virtual ~FWListMultipleModels();

      // ---------- const member functions ---------------------
      Bool_t CanEditMainColor() const;
      Bool_t SingleRnrState() const;
      ClassDef(FWListMultipleModels,0);

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void SetMainColor(Color_t iColor);
      void SetRnrState(Bool_t rnr);

      bool doSelection(bool);
   private:
      FWListMultipleModels(const FWListMultipleModels&); // stop default

      const FWListMultipleModels& operator=(const FWListMultipleModels&); // stop default

      // ---------- member data --------------------------------
      std::set<FWModelId> m_ids;
      Color_t m_color;

};


#endif
