
#ifndef Fireworks_Core_GUIEventFilter_h
#define Fireworks_Core_GUIEventFilter_h

#include <list>

#include "TGFrame.h"

#include "Fireworks/Core/interface/FWEventSelector.h"
#ifndef __CINT__
//#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"
#endif

class TGLabel;
class TGRadioButton;
class TGTextButton;
class CSGAction;
class FWCustomIconsButton;
class FWGUIEventSelector;
//class FWJobMetadataManager;
class CmsShowNavigator;
class FWConfiguration;

namespace fireworks
{
class Context;
}

class FWGUIEventFilter: public TGMainFrame
{
public:
   FWGUIEventFilter(CmsShowNavigator*);
   virtual ~FWGUIEventFilter();
   virtual void CloseWindow();
   
   void show(std::list<FWEventSelector*>* sels, int filterMode, int state);
   void reset();    

   std::list<FWGUIEventSelector*>& guiSelectors() { return m_guiSelectors; }
   
   void newEventEntry();
   void newTriggerEntry();
   void addSelector(FWEventSelector* sel);
   void deleteEntry(FWGUIEventSelector*);
   bool isOpen() { return m_isOpen; }
   void apply();
   void disableFilters();
   void setupDisableFilteringButton(bool);
   void checkApplyButton();
   void changeFilterMode();
   int  getFilterMode();
   void updateFilterStateLabel(int);
   /*
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);
   */
   Bool_t HandleKey(Event_t *event);
   ClassDef(FWGUIEventFilter, 0);
   
private:   
   static const int s_entryHeight = 21;
   
   int          m_origFilterMode;
   bool         m_isOpen;
   bool         m_filtersRemoved;
   
   std::list<FWGUIEventSelector*> m_guiSelectors;

   TGCompositeFrame*    m_eventSelectionFrameParent;
   TGCompositeFrame*    m_eventSelectionFrame;

   TGCompositeFrame*    m_triggerSelectionFrameParent;
   TGCompositeFrame*    m_triggerSelectionFrame;

   TGRadioButton*       m_rad1;
   TGRadioButton*       m_rad2;
   TGLabel*             m_stateLabel;
   TGTextButton*        m_applyBtn;
   TGTextButton*        m_disableFilteringBtn;
   FWCustomIconsButton* m_addBtn;

   CmsShowNavigator*    m_navigator;
};

#endif
