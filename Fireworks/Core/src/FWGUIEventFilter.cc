#include "TGPicture.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGButtonGroup.h"

#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

const TGPicture* FWGUIEventFilter::m_icon_add = 0;

FWGUIEventFilter::FWGUIEventFilter(const TGWindow* parent):
   TGTransientFrame(gClient->GetRoot(), parent, m_width+4, m_height),
   m_origOr(false),
   m_applyAction(0),
   m_finishEditAction(0),
   m_validator(0),
   m_selectionFrameParent(0),
   m_selectionFrame(0),
   m_btnGroup(false)
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

   TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 170, 22, kFixedSize);
   selH->AddFrame(cfr, new TGLayoutHints(kLHintsRight));
   cfr->AddFrame(new TGLabel(cfr, "Comment:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom));

   //-------------------- adding new selection

   if (!m_icon_add)
      m_icon_add = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"plus-sign.png");

   TGHorizontalFrame* addBtnFrame = new TGHorizontalFrame(v1, 29, 24, kFixedSize);
   v1->AddFrame(addBtnFrame, new TGLayoutHints(kLHintsRight));
   TGPictureButton* addButton = new TGPictureButton(addBtnFrame, m_icon_add);
   addBtnFrame->AddFrame(addButton, new TGLayoutHints(kLHintsRight|kLHintsExpandX|kLHintsExpandY, 0, 4, 1, 1));
   TQObject::Connect(addButton, "Clicked()", "FWGUIEventFilter",  this, "newEntry()");

   //-------------------- external actions
   m_finishEditAction = new CSGAction(this, "Finish");

   TGHorizontalFrame* btnFrame = new TGHorizontalFrame(v1, m_width, 2*m_entryHeight, 0);
   v1->AddFrame(btnFrame, new TGLayoutHints(kLHintsExpandX|kLHintsBottom));

   m_applyAction = new CSGAction(this, "Apply");
   m_applyAction->createTextButton(btnFrame,new TGLayoutHints(kLHintsLeft, 4, 2, 2, 2) );

   TGTextButton* revert = new TGTextButton(btnFrame," Revert ");
   btnFrame->AddFrame(revert, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2));
   revert->Connect("Clicked()","FWGUIEventFilter", this, "revert()");

   TGTextButton* ok = new TGTextButton(btnFrame," OK ");
   btnFrame->AddFrame(ok, new TGLayoutHints(kLHintsRight, 2, 2, 2, 2));
   ok->Connect("Clicked()","FWGUIEventFilter", this, "filterOK()");

   TGTextButton* cancel = new TGTextButton(btnFrame," Cancel ");
   btnFrame->AddFrame(cancel, new TGLayoutHints(kLHintsRight, 4, 2, 2, 2));
   cancel->Connect("Clicked()","FWGUIEventFilter", this, "CloseWindow()");
}

//______________________________________________________________________________

void FWGUIEventFilter::CloseWindow()
{
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

void FWGUIEventFilter::revert()
{
   m_btnGroup->SetButton( m_origOr ? 1 : 2);

   std::list<FWEventSelector*> orig;
   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
   {
      if ((*i)->origSelector())
         orig.push_back((*i)->origSelector());
   }

   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
   { 
      m_selectionFrame->RemoveFrame(*i);
   }

   FWGUIEventSelector* gs;
   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
   { 
      gs = *i;
      delete gs;
   }
   m_guiSelectors.clear();


   // add
   for(std::list<FWEventSelector*>::iterator i = orig.begin(); i != orig.end(); ++i)
      addSelector(*i);

   MapSubwindows();
   Layout();
   gClient->NeedRedraw(this);
}

//______________________________________________________________________________
 
void FWGUIEventFilter::addSelector(FWEventSelector* sel)
{
   FWGUIEventSelector* es = new FWGUIEventSelector(m_selectionFrame, m_validator, sel);
   m_selectionFrame->AddFrame(es, new TGLayoutHints(kLHintsExpandX));
   TQObject::Connect(es, "removeSelector(FWGUIEventSelector*)", "FWGUIEventFilter",  this, "deleteEntry(FWGUIEventSelector*)");
   
   m_guiSelectors.push_back(es);
}

//______________________________________________________________________________

void FWGUIEventFilter::show( std::list<FWEventSelector*>* sels,  fwlite::Event* event, bool isLogicalOR)
{
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

void FWGUIEventFilter::filterOK()
{
   m_applyAction->activated();
   CloseWindow();
}
