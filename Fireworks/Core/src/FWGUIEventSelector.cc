#include "TGButton.h"
#include "TGLabel.h"
#include "TSystem.h"
#include "TGPicture.h"

#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

const TGPicture* FWGUIEventSelector::m_icon_delete = 0;

FWGUIEventSelector::FWGUIEventSelector(TGCompositeFrame* p, FWEventSelector* sel, FWHLTValidator* validator):
   TGHorizontalFrame(p)
{
   m_selector = sel;

   // -------------- expression

   FWGUIValidatingTextEntry* text1 = new FWGUIValidatingTextEntry(this, sel->selection.c_str());
   text1->setValidator(validator);
   text1->ChangeOptions(0);
   text1->Connect("TextChanged(char*)", "string",&sel->selection, "assign(char*)");
   AddFrame(text1, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 2,2,1,1));
    
   // -------------- comment

   TGCompositeFrame *cfr = new TGHorizontalFrame(this, 120, 22, kFixedSize);
   AddFrame(cfr,new TGLayoutHints( kLHintsNormal, 2, 2, 1, 1 ));
   FWGUIValidatingTextEntry* text2 = new FWGUIValidatingTextEntry(cfr, sel->title.c_str());
   text2->setValidator(validator);
   text2->ChangeOptions(0);
   text2->Connect("TextChanged(char*)", "string",&sel->title, "assign(char*)");
   cfr->AddFrame(text2);
  
   // ---------------- enable

   TGCheckButton* checkButton = new TGCheckButton(this,"");
   checkButton->SetToolTipText("Enable/disable the selection");
   checkButton->SetOn(sel->enabled);
   AddFrame(checkButton, new TGLayoutHints(kLHintsNormal |  kLHintsCenterY , 2, 2, 0, 0));

   // ---------------- delete 
   if (!m_icon_delete)
      m_icon_delete = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"delete.png");
   TGPictureButton* deleteButton = new TGPictureButton(this, m_icon_delete);
   AddFrame(deleteButton, new TGLayoutHints(kLHintsRight, 2,2,1,1));
   TQObject::Connect(deleteButton, "Clicked()", "FWGUIEventSelector",  this, "deleteAction()");
}

//______________________________________________________________________________
void FWGUIEventSelector::removeSelector(FWGUIEventSelector* s)
{
   Emit("removeSelector(FWGUIEventSelector*)", (Long_t)s);
}

//______________________________________________________________________________
void FWGUIEventSelector::deleteAction()
{
   removeSelector(this);
}
