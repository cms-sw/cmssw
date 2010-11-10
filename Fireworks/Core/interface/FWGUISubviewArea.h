#ifndef Fireworks_Core_FWGUISubviewArea_h
#define Fireworks_Core_FWGUISubviewArea_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUISubviewArea
//
/**\class FWGUISubviewArea FWGUISubviewArea.h Fireworks/Core/interface/FWGUISubviewArea.h

   Description: Manages the GUI area where a sub Subview is displayed

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 15 14:13:29 EST 2008
// $Id: FWGUISubviewArea.h,v 1.25 2010/03/26 20:20:20 matevz Exp $
//

#include "TGFrame.h"
#include <sigc++/signal.h>
#include <string>

// forward declarations
class TGPictureButton;
class TGLabel;
class TEveCompositeFrame;
class TEveWindow;

class FWGUISubviewArea : public TGHorizontalFrame
{
public:
   FWGUISubviewArea(TEveCompositeFrame* ef, TGCompositeFrame* parent, Int_t height);
   virtual ~FWGUISubviewArea();

   // ---------- const member functions ---------------------

   bool         isSelected() const;

   // ---------- static member functions --------------------
   static const TGPicture * swapIcon();
   static const TGPicture * swapDisabledIcon();
   static const TGPicture * undockIcon();
   static const TGPicture * dockIcon();
   static const TGPicture * undockDisabledIcon();
   static const TGPicture * closeIcon();
   static const TGPicture * closeDisabledIcon();
   static const TGPicture * infoIcon();
   static const TGPicture * infoDisabledIcon();

   // ---------- member functions ---------------------------
   void unselect();
   void setSwapIcon(bool);
   void swap();
   void destroy();
   void undock();
   void dock();

   void selectButtonToggle();

   sigc::signal<void, FWGUISubviewArea*> swap_;
   sigc::signal<void, FWGUISubviewArea*> goingToBeDestroyed_;
   sigc::signal<void, FWGUISubviewArea*> selected_;
   sigc::signal<void, FWGUISubviewArea*> unselected_;

   void setInfoButton(bool downp);

   TEveWindow* getEveWindow();

   static FWGUISubviewArea* getToolBarFromWindow(TEveWindow*);

private:
   FWGUISubviewArea(const FWGUISubviewArea&);    // stop default
   const FWGUISubviewArea& operator=(const FWGUISubviewArea&);    // stop default

   // ---------- member data --------------------------------
   TEveCompositeFrame*  m_frame;

   TGPictureButton* m_swapButton;
   TGPictureButton* m_undockButton;
   TGPictureButton* m_dockButton;
   TGPictureButton* m_closeButton;
   TGPictureButton* m_infoButton;

   ClassDef(FWGUISubviewArea, 1);
};

#endif
