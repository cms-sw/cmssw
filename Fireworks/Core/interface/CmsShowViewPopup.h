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
// $Id: CmsShowViewPopup.h,v 1.9 2009/06/28 19:54:45 amraktad Exp $
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>

#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

// forward declarations
class TGLabel;
class TGTextButton;
class TGButton;
class TGFrame;
class TEveWindow;

class FWParameterSetterBase;
class FWViewBase;
class FWColorManager;

class CmsShowViewPopup : public TGTransientFrame, public FWParameterSetterEditorBase
{

public:
   CmsShowViewPopup(const TGWindow* p = 0, UInt_t w = 200, UInt_t h = 200, FWColorManager* cm=0, TEveWindow* ew = 0);
   virtual ~CmsShowViewPopup();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void CloseWindow();
   virtual void MapWindow();
   virtual void UnmapWindow();

   bool mapped() { return m_mapped; }

   void reset(TEveWindow* ew);

   void saveImage();
   void changeBackground();
   void backgroundColorWasChanged();

   TEveWindow* GetEveWindow() const { return m_eveWindow; }

   sigc::signal<void> closed_;

private:
   CmsShowViewPopup(const CmsShowViewPopup&);    // stop default

   const CmsShowViewPopup& operator=(const CmsShowViewPopup&);    // stop default

   // ---------- member data --------------------------------
   TGLabel* m_viewLabel;
   TGTextButton* m_removeButton;
   TGCompositeFrame* m_viewContentFrame;
   TGButton* m_saveImageButton;
   TGTextButton* m_changeBackground;
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
   FWColorManager* m_colorManager;
   TEveWindow* m_eveWindow;

   bool  m_mapped;
};


#endif
