#include <list>

#include "TGFrame.h"

#include "Fireworks/Core/interface/FWEventSelector.h"
#ifndef __CINT__
//#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"
#endif

class TGLabel;
class TGButtonGroup;
class TGTextButton;
class CSGAction;
class FWCustomIconsButton;
class FWGUIEventSelector;
class FWJobMetadataManager;

class FWGUIEventFilter: public TGTransientFrame
#ifndef __CINT__
                        ,CSGActionSupervisor
#endif
{
public:
   FWGUIEventFilter(const TGWindow*, FWJobMetadataManager*);
   virtual ~FWGUIEventFilter();
   virtual void CloseWindow();
   
   void show(std::list<FWEventSelector*>* sels, int filterMode, int state);
   void reset();    

   CSGAction* m_applyAction;   
   CSGAction* m_filterDisableAction;     
   CSGAction* m_finishEditAction; 

   std::list<FWGUIEventSelector*>& guiSelectors() { return m_guiSelectors; }
   
   void newEventEntry();
   void newTriggerEntry();
   void addSelector(FWEventSelector* sel);
   void deleteEntry(FWGUIEventSelector*);
   bool isOpen() { return m_isOpen; }
   void apply();
   void checkApplyButton();
   void changeFilterMode(Int_t);
   int  getFilterMode();
   void updateFilterStateLabel(int);

   ClassDef(FWGUIEventFilter, 0);
   
private:   
   static const int m_entryHeight = 21;
   static const int m_width       = 530;
   static const int m_height      = 320;
   
   int               m_origFilterMode;
   bool              m_isOpen;
   bool              m_filtersRemoved;
   
   std::list<FWGUIEventSelector*> m_guiSelectors;

   TGCompositeFrame*    m_eventSelectionFrameParent;
   TGCompositeFrame*    m_eventSelectionFrame;

   TGCompositeFrame*    m_triggerSelectionFrameParent;
   TGCompositeFrame*    m_triggerSelectionFrame;

   TGButtonGroup*       m_btnGroup;
   TGLabel*             m_stateLabel;
   TGTextButton*        m_applyBtn;
   FWCustomIconsButton* m_addBtn;

   FWJobMetadataManager* m_metadataManager;
};

