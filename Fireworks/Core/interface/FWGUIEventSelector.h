#ifndef Fireworks_Core_FWGUIEventSelector_h
#define Fireworks_Core_FWGUIEventSelector_h

#include "TGFrame.h"

class FWEventSelector;
class FWHLTValidator;
class TGPicture;


class FWGUIEventSelector : public TGHorizontalFrame {
public:
   FWGUIEventSelector(TGCompositeFrame* p, FWEventSelector* sel, FWHLTValidator* es);
   virtual ~FWGUIEventSelector() {}

   void deleteAction();
   void removeSelector(FWGUIEventSelector*); // *SIGNAL*
   FWEventSelector* getSelector() { return m_selector; }
 
private:

   FWGUIEventSelector(const FWGUIEventSelector&); // stop default
   const FWGUIEventSelector& operator=(const FWGUIEventSelector&); // stop default

   static const TGPicture* m_icon_delete;
   FWEventSelector*        m_selector;

   ClassDef(FWGUIEventSelector, 0); // Manager for EVE windows.
};

#endif

