#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"


FWGUIEventSelector::FWGUIEventSelector(TGCompositeFrame* p, FWHLTValidator* validator, FWEventSelector* sel):
   TGHorizontalFrame(p),
   m_guiSelector(0),
   m_origSelector(0),
   m_text1(0),
   m_text2(0),
   m_enableBtn(0),
   m_deleteBtn(0)
{
   m_origSelector = sel;
   m_guiSelector = new FWEventSelector();
   if (m_origSelector) *m_guiSelector = *m_origSelector;
   
   // -------------- expression

   m_text1 = new FWGUIValidatingTextEntry(this, m_guiSelector->m_expression.c_str());
   m_text1->setValidator(validator);
   m_text1->ChangeOptions(0);
   m_text1->Connect("TextChanged(char*)", "string",  &m_guiSelector->m_expression, "assign(char*)");
   AddFrame(m_text1, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 2,2,1,1));
    
   // -------------- comment

   TGCompositeFrame *cfr = new TGHorizontalFrame(this, 120, 22, kFixedSize);
   AddFrame(cfr,new TGLayoutHints( kLHintsNormal, 2, 2, 1, 1 ));
   m_text2 = new FWGUIValidatingTextEntry(cfr, m_guiSelector->m_description.c_str());
   m_text2->setValidator(validator);
   m_text2->ChangeOptions(0);
   m_text2->Connect("TextChanged(char*)", "string", &m_guiSelector->m_description, "assign(char*)");
   cfr->AddFrame(m_text2, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 2,2,1,1));
  
   // ---------------- enable

   m_enableBtn = new TGCheckButton(this,"");
   m_enableBtn->SetToolTipText("Enable/disable the selection");
   m_enableBtn->SetOn(m_guiSelector->m_enabled);
   m_enableBtn->Connect("Toggled(bool)","FWGUIEventSelector", this, "enableCallback(bool)");
   AddFrame(m_enableBtn, new TGLayoutHints(kLHintsNormal |  kLHintsCenterY , 2, 2, 0, 0));

   // ---------------- delete 
   m_deleteBtn = new FWCustomIconsButton(this, fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"delete.png"),
                                               fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"delete-over.png"),
                                               fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"delete-disabled.png"));
   AddFrame(m_deleteBtn, new TGLayoutHints(kLHintsRight|kLHintsCenterY, 2,2,0, 0));
   TQObject::Connect(m_deleteBtn, "Clicked()", "FWGUIEventSelector",  this, "deleteCallback()");

}
//____________________________________________________________________________
FWGUIEventSelector::~FWGUIEventSelector()
{
   delete m_guiSelector;
}


//____________________________________________________________________________
void FWGUIEventSelector::setActive(bool x)
{
   m_text1->SetEnabled(x);
   m_text2->SetEnabled(x);
   m_enableBtn->SetEnabled(x);
   m_deleteBtn->SetEnabled(x);
}

//____________________________________________________________________________
void FWGUIEventSelector::enableCallback(bool x)
{
   m_guiSelector->m_enabled = x;
}

//______________________________________________________________________________
void FWGUIEventSelector::removeSelector(FWGUIEventSelector* s)
{
   Emit("removeSelector(FWGUIEventSelector*)", (Long_t)this);
}

//______________________________________________________________________________
void FWGUIEventSelector::deleteCallback()
{
   removeSelector(this);
}
