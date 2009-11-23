#include <list>

#include "TGFrame.h"

#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class FWGUIEventSelector;
class TGButtonGroup;
class CSGAction;
class FWCustomIconsButton;

class FWGUIEventFilter: public TGTransientFrame,
                               CSGActionSupervisor
{
public:
   FWGUIEventFilter(const TGWindow* parent);
   virtual void CloseWindow();
   
   void show(std::list<FWEventSelector*>* sels,  fwlite::Event* event, bool isLogicalOR);
   
   CSGAction* m_applyAction;   
   CSGAction* m_toggleEnableAction;     
   CSGAction* m_finishEditAction; 

   std::list<FWGUIEventSelector*>& guiSelectors() { return m_guiSelectors; }
   bool isLogicalOR();
   
   void newEntry();
   void deleteEntry(FWGUIEventSelector*);
   void addSelector(FWEventSelector* sel);
   bool isOpen() { return m_isOpen; }
   void setActive(bool);
   bool isActive() const { return m_active; }
   
private:   
   static const int m_entryHeight = 20;
   static const int m_width       = 500;
   static const int m_height      = 300;
   
   bool              m_origOr;
   bool              m_isOpen;
   bool              m_active;
   
   std::list<FWGUIEventSelector*> m_guiSelectors;
   FWHLTValidator*      m_validator;
   TGCompositeFrame*    m_selectionFrameParent;
   TGCompositeFrame*    m_selectionFrame;
   TGButtonGroup*       m_btnGroup;   
   FWCustomIconsButton* m_addBtn;
};

