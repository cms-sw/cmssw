#ifndef Fireworks_Core_FWGUIEventSelector_h
#define Fireworks_Core_FWGUIEventSelector_h

#include "TGFrame.h"

class TGLabel;

class FWEventSelector;
class FWHLTValidator;
class FWCustomIconsButton;
class TGCheckButton;
class FWGUIValidatingTextEntry;


class FWGUIEventSelector : public TGHorizontalFrame {
public:
   FWGUIEventSelector(TGCompositeFrame* p, FWHLTValidator* es, FWEventSelector* sel);
   virtual ~FWGUIEventSelector();

   void deleteCallback();
   void enableCallback(bool);
   void expressionCallback(char*);
   void updateNEvents();

   
   FWEventSelector* guiSelector()  { return m_guiSelector;  }
   FWEventSelector* origSelector() { return m_origSelector; }
   void setOrigSelector(FWEventSelector* s) { m_origSelector = s; }

   void removeSelector(FWGUIEventSelector*);  // *SIGNAL*
   void selectorChanged();                    // *SIGNAL*

private:

   FWGUIEventSelector(const FWGUIEventSelector&); // stop default
   const FWGUIEventSelector& operator=(const FWGUIEventSelector&); // stop default
   
   FWEventSelector*   m_guiSelector;
   FWEventSelector*   m_origSelector;
   
   FWGUIValidatingTextEntry* m_text1;
   FWGUIValidatingTextEntry* m_text2;
   TGCheckButton*            m_enableBtn;
   FWCustomIconsButton*      m_deleteBtn;
   TGLabel*                  m_nEvents;
   
   ClassDef(FWGUIEventSelector, 0); // Manager for EVE windows.
};

#endif

