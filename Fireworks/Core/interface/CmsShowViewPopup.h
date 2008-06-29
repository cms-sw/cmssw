#ifndef Fireworks_Core_CmsShowViewPopup_h
#define Fireworks_Core_CmsShowViewPopup_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowViewPopup
// 
/**\class CmsShowViewPopup CmsShowViewPopup.h Fireworks/Core/interface/CmsShowViewPopup.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Jun 25 15:15:12 EDT 2008
// $Id$
//

// system include files
#include <vector>
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

// forward declarations
class FWViewBase;
class TGLabel;
class TGTextButton;
class TGFrame;
class FWParameterSetterBase;

class CmsShowViewPopup : public TGMainFrame, public FWParameterSetterEditorBase
{

   public:
      CmsShowViewPopup(const TGWindow* p = 0, UInt_t w = 0, UInt_t h = 0, FWViewBase* v = 0);
      virtual ~CmsShowViewPopup();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void reset(FWViewBase* iView);
      void removeView();
 
   private:
      CmsShowViewPopup(const CmsShowViewPopup&); // stop default

      const CmsShowViewPopup& operator=(const CmsShowViewPopup&); // stop default

      // ---------- member data --------------------------------
      TGLabel* m_viewLabel;
      TGTextButton* m_removeButton;
      TGCompositeFrame* m_viewContentFrame;
      FWViewBase* m_view;
      std::vector<FWParameterSetterBase*> m_setters;
};


#endif
