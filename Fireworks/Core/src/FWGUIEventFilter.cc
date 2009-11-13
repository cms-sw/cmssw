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
   m_sels(0),
   m_validator(0),
   m_selectionFrameParent(0),
   m_selectionFrame(0),
   m_orBtn(0)
{  
   TGVerticalFrame* v1 = new TGVerticalFrame(this);
   AddFrame(v1, new TGLayoutHints(kLHintsExpandX |kLHintsExpandY));

   //-------------------- logical operations

   TGHorizontalFrame* headerFrame = new TGHorizontalFrame(v1, m_width, 2*m_entryHeight, 0);
   v1->AddFrame(headerFrame, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 1, 1, 1, 1));

   TGButtonGroup* cont = new TGButtonGroup(headerFrame, "Outputs of enabled selectors are combined as the logical:");
   m_orBtn = new TGRadioButton(cont, "OR", 1);
   new TGRadioButton(cont, "AND", 2);
   headerFrame->AddFrame(cont, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
  
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

   TGHorizontalFrame* btnFrame = new TGHorizontalFrame(v1, m_width, 2*m_entryHeight, 0);
   v1->AddFrame(btnFrame, new TGLayoutHints(kLHintsExpandX|kLHintsBottom));

   m_applyAction = new CSGAction(this, "Apply");
   m_applyAction->createTextButton(btnFrame,new TGLayoutHints(kLHintsRight, 1, 1, 1, 1) );

   TGTextButton* ok = new TGTextButton(btnFrame," OK ");
   btnFrame->AddFrame(ok, new TGLayoutHints(kLHintsRight, 1, 1, 1, 1));
   ok->Connect("Clicked()","FWGUIEventFilter", this, "filterOK()");

   TGTextButton* cancel = new TGTextButton(btnFrame," Cancel ");
   btnFrame->AddFrame(cancel, new TGLayoutHints(kLHintsRight, 1, 1, 1, 1));
   cancel->Connect("Clicked()","FWGUIEventFilter", this, "CloseWindow()");
}

void FWGUIEventFilter::CloseWindow()
{
   if (m_selectionFrame) {
      m_selectionFrameParent->RemoveFrame(m_selectionFrame);
      m_selectionFrame = 0;

      delete m_validator;
      m_validator = 0;
   }

   UnmapWindow();
}

void FWGUIEventFilter::filterOK()
{
   m_applyAction->activated();
   CloseWindow();
}
 
void FWGUIEventFilter::show( std::vector<FWEventSelector*>* sels,  fwlite::Event& event, bool isLogicalOR)
{
   m_sels = sels;
   m_validator = new FWHLTValidator(event);
   m_orBtn->SetOn(isLogicalOR, kFALSE);

   assert(m_selectionFrame == 0);
   m_selectionFrame = new TGVerticalFrame(m_selectionFrameParent);
   m_selectionFrameParent->AddFrame(m_selectionFrame,  new TGLayoutHints(kLHintsExpandX));

   for(std::vector<FWEventSelector*>::iterator i = m_sels->begin(); i != m_sels->end(); ++i)
      addSelector(*i);

   MapSubwindows();
   Layout();
   MapWindow();
}

void FWGUIEventFilter::addSelector(FWEventSelector* sel)
{
   FWGUIEventSelector* es = new FWGUIEventSelector(m_selectionFrame, sel, m_validator);
   m_selectionFrame->AddFrame(es, new TGLayoutHints(kLHintsExpandX));

   TQObject::Connect(es, "removeSelector(FWGUIEventSelector*)", "FWGUIEventFilter",  this, "deleteEntry(FWGUIEventSelector*)");
}

void FWGUIEventFilter::deleteEntry(FWGUIEventSelector* sel)
{
   sel->getSelector()->removed = true;
   m_selectionFrame->RemoveFrame(sel);
   Layout();
   gClient->NeedRedraw(this);
}

void FWGUIEventFilter::newEntry()
{
   FWEventSelector* sel = new  FWEventSelector();
   m_sels->push_back(sel);
   addSelector(sel);
   MapSubwindows();
   Layout();
}

bool FWGUIEventFilter::isLogicalOR()
{
   return m_orBtn->GetState();
}

void FWGUIEventFilter::dump(const char* text){
  std::cout << "Text changed: " << text << std::endl;
  
  for(std::vector<FWEventSelector*>::iterator sel = m_sels->begin();
      sel != m_sels->end(); ++sel)
    std::cout << "\t" << (*sel)->enabled << "\t " << (*sel)->selection << "\t" << (*sel)->title<< 
      "\t " << (*sel)->removed << std::endl;
}
