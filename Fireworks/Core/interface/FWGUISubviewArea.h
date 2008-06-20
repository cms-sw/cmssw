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
// $Id: FWGUISubviewArea.h,v 1.2 2008/03/16 15:23:49 chrjones Exp $
//

// system include files
#include "TGFrame.h"
#include <sigc++/signal.h>
// user include files

// forward declarations
class TGSplitFrame;
class TGButton;

class FWGUISubviewArea : public TGHorizontalFrame
{

   public:
      FWGUISubviewArea(unsigned int iIndex, const TGSplitFrame *iParent,TGSplitFrame*);
      virtual ~FWGUISubviewArea();

      // ---------- const member functions ---------------------
      //index says which sub area this occupies [used for configuration saving]
      unsigned int index() const {
         return m_index;
      }
   
      // ---------- static member functions --------------------
      static const TGPicture * swapIcon();
   
      // ---------- member functions ---------------------------
      void swapToBigView();
    
      void setIndex(unsigned int iIndex);   
      sigc::signal<void,unsigned int> swappedToBigView_;
   private:
      FWGUISubviewArea(const FWGUISubviewArea&); // stop default

      const FWGUISubviewArea& operator=(const FWGUISubviewArea&); // stop default

      // ---------- member data --------------------------------
      TGSplitFrame* m_mainSplit;
      unsigned int m_index;
      TGButton* m_swapButton;
};


#endif
