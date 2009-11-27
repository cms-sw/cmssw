#include <list>

#include "TGFrame.h"

#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class FWGUIEventSelector;
class TGButtonGroup;
class TGTextButton;
class CSGAction;
class FWCustomIconsButton;

class FWGUIEventFilter: public TGTransientFrame,
                               CSGActionSupervisor
{
public:
   FWGUIEventFilter(const TGWindow* parent);
   virtual void CloseWindow();
   
   void show(std::list<FWEventSelector*>* sels,  fwlite::Event* event, int filterMode);
   
   CSGAction* m_applyAction;   
   CSGAction* m_filterDisableAction;     
   CSGAction* m_finishEditAction; 

   std::list<FWGUIEventSelector*>& guiSelectors() { return m_guiSelectors; }
   
   void newEntry();
   void addSelector(FWEventSelector* sel);
   void deleteEntry(FWGUIEventSelector*);
   bool isOpen() { return m_isOpen; }
   void apply();
   void checkApplyButton();
   void changeFilterMode(Int_t);
   int  getFilterMode();
   
private:   
   static const int m_entryHeight = 20;
   static const int m_width       = 500;
   static const int m_height      = 300;
   
   int               m_origFilterMode;
   bool              m_isOpen;
   bool              m_filtersRemoved;
   
   std::list<FWGUIEventSelector*> m_guiSelectors;
   FWHLTValidator*      m_validator;
   TGCompositeFrame*    m_selectionFrameParent;
   TGCompositeFrame*    m_selectionFrame;
   TGButtonGroup*       m_btnGroup;
   TGTextButton*        m_applyBtn;
   FWCustomIconsButton* m_addBtn;
};

