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
// $Id: FWGUISubviewArea.h,v 1.11 2009/01/23 21:35:41 amraktad Exp $
//

#include "TGFrame.h"
#include <sigc++/signal.h>
#include <string>

// forward declarations
class TGButton;
class TGLabel;
class TEveCompositeFrame;
class TEveWindow;

class FWGUISubviewArea : public TGHorizontalFrame
{
public:
   FWGUISubviewArea(unsigned int idx, TEveCompositeFrame* eveWindow);
   virtual ~FWGUISubviewArea();

   // ---------- const member functions ---------------------
   //index says which sub area this occupies [used for configuration saving]
   unsigned int index()      const { return m_index; }
   bool         isSelected() const;

   // ---------- static member functions --------------------
   static const TGPicture * swapIcon();
   static const TGPicture * swapDisabledIcon();
   static const TGPicture * undockIcon();
   static const TGPicture * undockDisabledIcon();
   static const TGPicture * closeIcon();
   static const TGPicture * closeDisabledIcon();
   static const TGPicture * infoIcon();
   static const TGPicture * infoDisabledIcon();

   // ---------- member functions ---------------------------
   void setIndex(unsigned int iIndex) { m_index = iIndex; }

   void unselect();
   void swapWithCurrentView();
   void destroy();
   void undock();
   void undockTo(Int_t x, Int_t y, UInt_t width, UInt_t height);

   void selectButtonDown();
   void selectButtonUp();

   sigc::signal<void,unsigned int> swapWithCurrentView_;
   sigc::signal<void,unsigned int> goingToBeDestroyed_;
   sigc::signal<void,unsigned int> selected_;
   sigc::signal<void,unsigned int> unselected_;

   TEveWindow* getEveWindow();

private:
   FWGUISubviewArea(const FWGUISubviewArea&);    // stop default
   const FWGUISubviewArea& operator=(const FWGUISubviewArea&);    // stop default

   // ---------- member data --------------------------------
   unsigned int m_index;
   TEveCompositeFrame*  m_frame;

   TGPictureButton* m_swapButton;
   TGPictureButton* m_undockButton;
   TGPictureButton* m_closeButton;
   TGPictureButton* m_infoButton;
};

#endif
