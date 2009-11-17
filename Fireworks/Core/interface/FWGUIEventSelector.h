#ifndef Fireworks_Core_FWGUIEventSelector_h
#define Fireworks_Core_FWGUIEventSelector_h

#include "TGFrame.h"

class FWEventSelector;
class FWHLTValidator;
class TGTextButton;
class TGPicture;


class FWGUIEventSelector : public TGHorizontalFrame {
public:
   FWGUIEventSelector(TGCompositeFrame* p, FWHLTValidator* es, FWEventSelector* sel);
   virtual ~FWGUIEventSelector();

   void deleteCallback();
   void enableCallback(bool);
   void removeSelector(FWGUIEventSelector*); // *SIGNAL*
   
   FWEventSelector* guiSelector()  { return m_guiSelector;  }
   FWEventSelector* origSelector() { return m_origSelector; }
   
private:

   FWGUIEventSelector(const FWGUIEventSelector&); // stop default
   const FWGUIEventSelector& operator=(const FWGUIEventSelector&); // stop default

   static const TGPicture* m_icon_delete;
   
   FWEventSelector*   m_guiSelector;
   FWEventSelector*   m_origSelector;

   ClassDef(FWGUIEventSelector, 0); // Manager for EVE windows.
};

#endif

