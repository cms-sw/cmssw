#include "TGLabel.h"
#include "TGButtonGroup.h"

#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

FWGUIEventFilter::FWGUIEventFilter(const TGWindow* parent):
   TGTransientFrame(gClient->GetRoot(), parent, m_width+4, m_height),
   m_applyAction(0),
   m_toggleEnableAction(0),
   m_finishEditAction(0),

   m_origOr(false), 
   m_isOpen(false),
   m_active(true),

   m_validator(0),
   m_selectionFrameParent(0),
   m_selectionFrame(0),
   m_btnGroup(0),
   m_addBtn(0)
{  
   SetWindowName("Event Filters");

   TGVerticalFrame* v1 = new TGVerticalFrame(this);
   AddFrame(v1, new TGLayoutHints(kLHintsExpandX |kLHintsExpandY));

   //-------------------- logical operations

   TGHorizontalFrame* headerFrame = new TGHorizontalFrame(v1, m_width, 2*m_entryHeight, 0);
   v1->AddFrame(headerFrame, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 1, 1, 1, 1));

   m_btnGroup = new TGButtonGroup(headerFrame, "Outputs of enabled selectors are combined as the logical:");
   new TGRadioButton(m_btnGroup, "OR",  1);
   new TGRadioButton(m_btnGroup, "AND", 2);
   headerFrame->AddFrame(m_btnGroup, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
  
   //-------------------- selection header

   m_selectionFrameParent =  new TGVerticalFrame(v1, m_width, m_entryHeight, 0);
   v1->AddFrame(m_selectionFrameParent, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 2, 2, 0,0));
   
   // headers
   TGHorizontalFrame* selH = new TGHorizontalFrame(m_selectionFrameParent);
   m_selectionFrameParent->AddFrame(selH, new TGLayoutHints(kLHintsExpandX));

   selH->AddFrame(new TGLabel(selH, "Expression:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom , 2, 0, 0, 0 ));

   TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 162, 22, kFixedSize);
   selH->AddFrame(cfr, new TGLayoutHints(kLHintsRight));
   cfr->AddFrame(new TGLabel(cfr, "Comment:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom));

   //-------------------- adding new selection

   TGHorizontalFrame* addBtnFrame = new TGHorizontalFrame(v1);
   v1->AddFrame(addBtnFrame, new TGLayoutHints(kLHintsRight));
   
   m_addBtn = new FWCustomIconsButton(addBtnFrame, fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign.png"),
                                                   fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-over.png"),
                                                   fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-disabled.png"));
   
   addBtnFrame->AddFrame(m_addBtn, new TGLayoutHints(kLHintsRight|kLHintsExpandX|kLHintsExpandY, 0, 6, 4, 1));
   TQObject::Connect(m_addBtn, "Clicked()", "FWGUIEventFilter",  this, "newEntry()");

   //-------------------- external actions
   m_finishEditAction = new CSGAction(this, "Finish");

   TGHorizontalFrame* btnFrame = new TGHorizontalFrame(v1, 250, 30, kFixedSize);
   v1->AddFrame(btnFrame, new TGLayoutHints(kLHintsCenterX | kLHintsBottom , 2, 2, 2, 4));

   m_applyAction = new CSGAction(this, " Filter ");
   m_applyAction->createTextButton(btnFrame,new TGLayoutHints(kLHintsLeft, 4, 2, 2, 4) );
   
   m_toggleEnableAction = new CSGAction(this, " Toggle Filtering ");
   m_toggleEnableAction->createTextButton(btnFrame,new TGLayoutHints(kLHintsLeft, 4, 2, 2, 4) );   
   
   TGTextButton* cancel = new TGTextButton(btnFrame," Close ");
   btnFrame->AddFrame(cancel, new TGLayoutHints(kLHintsRight, 4, 4, 2, 4));
   cancel->Connect("Clicked()","FWGUIEventFilter", this, "CloseWindow()");
}

//______________________________________________________________________________

void FWGUIEventFilter::CloseWindow()
{
   m_isOpen = false;
   m_selectionFrameParent->RemoveFrame(m_selectionFrame);
   m_selectionFrame = 0;
   
   FWGUIEventSelector* gs;
   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
   { 
      gs = *i;
      delete gs;
   }

   m_guiSelectors.clear();   
      
   delete m_validator;
   m_validator = 0;
   
   UnmapWindow();
   m_finishEditAction->activated();
}

//______________________________________________________________________________
 
void FWGUIEventFilter::addSelector(FWEventSelector* sel)
{
   FWGUIEventSelector* es = new FWGUIEventSelector(m_selectionFrame, m_validator, sel);
   if (!m_active) es->setActive(false);
   m_selectionFrame->AddFrame(es, new TGLayoutHints(kLHintsExpandX));
   TQObject::Connect(es, "removeSelector(FWGUIEventSelector*)", "FWGUIEventFilter",  this, "deleteEntry(FWGUIEventSelector*)");
   
   m_guiSelectors.push_back(es);
}

//______________________________________________________________________________

void FWGUIEventFilter::show( std::list<FWEventSelector*>* sels,  fwlite::Event* event, bool isLogicalOR)
{
   m_isOpen = true;
   m_validator = new FWHLTValidator(*event);

   m_origOr = isLogicalOR;
   m_btnGroup->SetButton( m_origOr ? 1 : 2);

   assert(m_selectionFrame == 0);
   m_selectionFrame = new TGVerticalFrame(m_selectionFrameParent);
   m_selectionFrameParent->AddFrame(m_selectionFrame,  new TGLayoutHints(kLHintsExpandX));

   for(std::list<FWEventSelector*>::iterator i = sels->begin(); i != sels->end(); ++i)
      addSelector(*i);

   MapSubwindows();
   Layout();
   MapWindow();
}

//______________________________________________________________________________

void FWGUIEventFilter::deleteEntry(FWGUIEventSelector* sel)
{
   m_guiSelectors.remove(sel);
   
   m_selectionFrame->RemoveFrame(sel);
   Layout();
   gClient->NeedRedraw(this);
   
}

void FWGUIEventFilter::newEntry()
{
   addSelector(0);
   MapSubwindows();
   Layout();
}

bool FWGUIEventFilter::isLogicalOR()
{
   return m_btnGroup->GetButton(1)->GetState();
}

void FWGUIEventFilter::setActive(bool x)
{
   m_active = x;
   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
      (*i)->setActive(x);

   m_btnGroup->SetState(x);
   m_addBtn->SetEnabled(x);
}
