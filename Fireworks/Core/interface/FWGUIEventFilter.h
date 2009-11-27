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
private:
   static const TGPicture* m_icon_add;

   static const int m_entryHeight = 20;
   static const int m_width = 500;
   static const int m_height = 300;

   std::vector<FWEventSelector*>* m_sels;
   FWHLTValidator*   m_validator;

   TGCompositeFrame* m_selectionFrameParent;
   TGCompositeFrame* m_selectionFrame;
   TGTextButton*     m_orBtn;

   void addSelector(FWEventSelector* sel);

public:
   FWGUIEventFilter(const TGWindow* parent);
   virtual void CloseWindow();

   void show(std::vector<FWEventSelector*>* sels,  fwlite::Event& event, bool isLogicalOR);
   bool isLogicalOR();
   void filterOK();
   void dump(const char* text);
   void newEntry();
   void deleteEntry(FWGUIEventSelector*);
   CSGAction* m_applyAction;  
};

