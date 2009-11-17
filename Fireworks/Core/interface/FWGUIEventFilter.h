#include <string>
#include <vector>
#include <iostream>
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGActionSupervisor.h"

class TGVerticalFrame;
class TGFrame;
class TGTransientFrame;
class TGCompositeFrame;
class TGTextButton;
class FWGUIEventSelector;

class FWGUIEventFilter: public TGTransientFrame,
                        CSGActionSupervisor
{
public:
   FWGUIEventFilter(const TGWindow* parent);
   virtual void CloseWindow();
   
   void show(std::list<FWEventSelector*>* sels,  fwlite::Event* event, bool isLogicalOR);
   
   CSGAction* m_applyAction;     
   
   std::list<FWGUIEventSelector*>& guiSelectors() { return m_guiSelectors; }
   bool isLogicalOR();
   
   void filterOK();
   
   void newEntry();
   void deleteEntry(FWGUIEventSelector*);
   void addSelector(FWEventSelector* sel);
   
private:   
   static const TGPicture* m_icon_add;
   
   static const int m_entryHeight = 20;
   static const int m_width       = 500;
   static const int m_height      = 300;
   
   std::list<FWGUIEventSelector*> m_guiSelectors;
   FWHLTValidator*   m_validator;
   
   TGCompositeFrame* m_selectionFrameParent;
   TGCompositeFrame* m_selectionFrame;
   TGTextButton*     m_orBtn;
};

