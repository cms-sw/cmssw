#include "TGLabel.h"
#include "TGComboBox.h"

#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

FWGUIEventSelector::FWGUIEventSelector(TGCompositeFrame* p,
                                       FWEventSelector* sel,
                                       std::vector<std::string>& triggerProcessList)
    : TGHorizontalFrame(p),
      m_guiSelector(nullptr),
      m_origSelector(nullptr),
      m_text1(nullptr),
      m_text2(nullptr),
      m_enableBtn(nullptr),
      m_deleteBtn(nullptr),
      m_nEvents(nullptr),
      m_combo(nullptr),
      m_validator(nullptr) {
  m_origSelector = sel;
  m_guiSelector = new FWEventSelector(m_origSelector);

  if (!m_guiSelector->m_triggerProcess.empty()) {
    m_combo = new TGComboBox(this);
    int cnt = 0;
    int id = -1;
    for (std::vector<std::string>::iterator i = triggerProcessList.begin(); i != triggerProcessList.end(); ++i) {
      if (*i == sel->m_triggerProcess)
        id = cnt;
      m_combo->AddEntry((*i).c_str(), cnt++);
    }

    if (id < 0) {
      m_combo->AddEntry(sel->m_triggerProcess.c_str(), cnt);
      id = cnt;
    }
    m_combo->Select(id, false);
    m_combo->Resize(80, 21);
    AddFrame(m_combo, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 1));

    m_validator = new FWHLTValidator(m_guiSelector->m_triggerProcess);
    m_combo->Connect("Selected(const char*)", "FWGUIEventSelector", this, "triggerProcessCallback(const char*)");
  }

  // -------------- expression

  m_text1 = new FWGUIValidatingTextEntry(this, m_guiSelector->m_expression.c_str());
  m_text1->SetHeight(20);
  m_text1->setValidator(m_validator);
  m_text1->ChangeOptions(0);
  m_text1->Connect("TextChanged(char*)", "FWGUIEventSelector", this, "expressionCallback(char*)");
  AddFrame(m_text1, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 2, 2, 1, 1));

  // -------------- comment

  TGCompositeFrame* cfr = new TGHorizontalFrame(this, 120, 20, kFixedSize);
  AddFrame(cfr, new TGLayoutHints(kLHintsNormal, 2, 2, 1, 3));
  m_text2 = new FWGUIValidatingTextEntry(cfr, m_guiSelector->m_description.c_str());
  //   m_text2->SetHeight(21);
  m_text2->setValidator(m_validator);
  m_text2->ChangeOptions(0);
  m_text2->Connect("TextChanged(char*)", "string", &m_guiSelector->m_description, "assign(char*)");
  cfr->AddFrame(m_text2, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 2, 2, 0, 0));

  // ---------------- selection info

  TGHorizontalFrame* labFrame = new TGHorizontalFrame(this, 60, 22, kFixedSize);
  AddFrame(labFrame, new TGLayoutHints(kLHintsBottom, 2, 2, 2, 2));
  m_nEvents = new TGLabel(labFrame, "");
  m_nEvents->SetTextJustify(kTextLeft);
  labFrame->AddFrame(m_nEvents, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX));
  updateNEvents();

  // ---------------- enable

  m_enableBtn = new TGCheckButton(this, "");
  m_enableBtn->SetToolTipText("Enable/disable the selection");
  m_enableBtn->SetOn(m_guiSelector->m_enabled);
  m_enableBtn->Connect("Toggled(bool)", "FWGUIEventSelector", this, "enableCallback(bool)");
  AddFrame(m_enableBtn, new TGLayoutHints(kLHintsNormal | kLHintsCenterY, 2, 2, 0, 0));

  // ---------------- delete
  m_deleteBtn = new FWCustomIconsButton(this,
                                        fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "delete.png"),
                                        fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "delete-over.png"),
                                        fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "delete-disabled.png"));
  AddFrame(m_deleteBtn, new TGLayoutHints(kLHintsRight | kLHintsCenterY, 2, 2, 0, 0));
  TQObject::Connect(m_deleteBtn, "Clicked()", "FWGUIEventSelector", this, "deleteCallback()");
}
//____________________________________________________________________________
FWGUIEventSelector::~FWGUIEventSelector() {
  delete m_guiSelector;
  delete m_validator;
}

//____________________________________________________________________________
void FWGUIEventSelector::enableCallback(bool x) {
  m_guiSelector->m_enabled = x;
  selectorChanged();
}

//______________________________________________________________________________
void FWGUIEventSelector::removeSelector(FWGUIEventSelector* s) {
  Emit("removeSelector(FWGUIEventSelector*)", (Long_t)s);
}

//______________________________________________________________________________
void FWGUIEventSelector::deleteCallback() { removeSelector(this); }

//______________________________________________________________________________
void FWGUIEventSelector::triggerProcessCallback(const char* txt) {
  m_guiSelector->m_triggerProcess = txt;
  m_validator->setProcess(txt);
  selectorChanged();
}

//______________________________________________________________________________
void FWGUIEventSelector::expressionCallback(char* txt) {
  m_guiSelector->m_expression = txt;
  selectorChanged();
}

//______________________________________________________________________________
void FWGUIEventSelector::selectorChanged() {
  if (m_origSelector)
    m_origSelector->m_updated = false;
  Emit("selectorChanged()");
}

//______________________________________________________________________________
void FWGUIEventSelector::updateNEvents() {
  if (m_origSelector && m_origSelector->m_updated) {
    m_nEvents->Enable();
    const char* txtInfo = Form("%3d", m_origSelector ? m_origSelector->m_selected : -1);
    m_nEvents->SetText(txtInfo);
  } else {
    m_nEvents->Disable();
    m_nEvents->SetText("no check");
  }
}
