#include "TGFrame.h"
#include "TGPicture.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGButton.h"
#include "TGCanvas.h"
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"
#include "Fireworks/Core/interface/FWEventSelector.h"
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "TSystem.h"

FWGUIEventFilter::FWGUIEventFilter(std::vector<FWEventSelector>& sels, bool& junction):
  TGTransientFrame(gClient->GetRoot(), gClient->GetRoot(),m_width,m_height),
  m_sels(sels),m_globalOR(junction),m_haveNewEntry(false)
{
  // SetCleanup(kDeepCleanup);
  TGHorizontalFrame* labels = new TGHorizontalFrame(this, m_width, 2*m_entryHeight, 0);
  AddFrame(labels, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 5,5,5,5));

  TGLabel* label1 = new TGLabel(labels," Outputs of enabled selectors are combined as the logical: ");
  labels->AddFrame(label1, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 0,0,0,0));
  m_junctionWidget = new TGTextButton(labels," OR  ");
  labels->AddFrame(m_junctionWidget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 0,0,0,0));
  m_junctionWidget->Connect("Clicked()","FWGUIEventFilter", this, "junctionChanged()");
  junctionUpdate();

  TGCanvas* frame = new TGCanvas(this,m_width,m_height-2*m_entryHeight);
  AddFrame(frame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 0,0,0,0));

  m_mainFrame = new TGHorizontalFrame(frame->GetViewPort(), m_width, m_entryHeight, 0);
  frame->SetContainer(m_mainFrame);
  
  m_columns.push_back(new TGVerticalFrame(m_mainFrame));
  m_mainFrame->AddFrame(m_columns.back(), new TGLayoutHints(kLHintsLeft, 0,0,0,0));
  m_cells.push_back(std::vector<TGFrame*>());
  m_cells.back().push_back(new TGLabel(m_columns.back(),"Enable"));
  m_columns.back()->AddFrame(m_cells.back().back(), new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  
  m_columns.push_back(new TGVerticalFrame(m_mainFrame));
  m_mainFrame->AddFrame(m_columns.back(), new TGLayoutHints(kLHintsCenterX | kLHintsExpandX, 0,0,0,0));
  m_cells.push_back(std::vector<TGFrame*>());
  m_cells.back().push_back(new TGLabel(m_columns.back(),"Selection Expression"));
  m_columns.back()->AddFrame(m_cells.back().back(), new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  
  m_columns.push_back(new TGVerticalFrame(m_mainFrame));
  m_mainFrame->AddFrame(m_columns.back(), new TGLayoutHints(kLHintsCenterX | kLHintsExpandX, 0,0,0,0));
  m_cells.push_back(std::vector<TGFrame*>());
  m_cells.back().push_back(new TGLabel(m_columns.back(),"Comment"));
  m_columns.back()->AddFrame(m_cells.back().back(), new TGLayoutHints(kLHintsCenterX,2,2,2,2));

  m_columns.push_back(new TGVerticalFrame(m_mainFrame));
  m_mainFrame->AddFrame(m_columns.back(), new TGLayoutHints(kLHintsRight, 0,0,0,0));
  m_cells.push_back(std::vector<TGFrame*>());
  m_cells.back().push_back(new TGLabel(m_columns.back(),"Delete"));
  m_columns.back()->AddFrame(m_cells.back().back(), new TGLayoutHints(kLHintsCenterX,2,2,2,2));
}

void FWGUIEventFilter::addSelector(FWEventSelector& sel){
  if (sel.removed){
    m_cells[0].push_back(0);
    m_cells[1].push_back(0);
    m_cells[2].push_back(0);
    m_cells[3].push_back(0);
    return;
  }
  static const TGPicture* s_icon = 0;
  if(0== s_icon) {
    const char* cmspath = gSystem->Getenv("CMSSW_BASE");
    if(0 == cmspath) {
      throw std::runtime_error("CMSSW_BASE environment variable not set");
    }
    s_icon = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"delete.png");
  }
  
  TGCheckButton* checkButton = new TGCheckButton(m_columns[0],"");
  checkButton->SetToolTipText("Enable/disable the selection");
  checkButton->SetOn(sel.enabled);
  m_columns[0]->AddFrame(checkButton, new TGLayoutHints(kLHintsCenterX,2,2,4,3));
  checkButton->Connect("Toggled(bool)","FWEventSelector", &sel, "enable(bool)");
  m_cells[0].push_back(checkButton);
    
  TGTextEntry* text1 = new TGTextEntry(m_columns[1], sel.selection.c_str());
  text1->ChangeOptions(0);
  m_columns[1]->AddFrame(text1, new TGLayoutHints(kLHintsCenterX | kLHintsExpandX, 1,1,1,1));
  text1->Connect("TextChanged(char*)", "string",&sel.selection, "assign(char*)");
  m_cells[1].push_back(text1);
    
  TGTextEntry* text2 = new TGTextEntry(m_columns[2], sel.title.c_str());
  text2->ChangeOptions(0);
  m_columns[2]->AddFrame(text2, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 1,1,1,1));
  text2->Connect("TextChanged(char*)", "string",&sel.title, "assign(char*)");
  m_cells[2].push_back(text2);
  
  TGPictureButton* button = new TGPictureButton(m_columns[3], s_icon);
  button->ChangeOptions(0);
  m_columns[3]->AddFrame(button, new TGLayoutHints(kLHintsCenterX, 1,1,1,1));
  button->Connect("Clicked()","FWEventSelector", &sel, "remove()");
  button->Connect("Clicked()","FWGUIEventFilter", this, "update()");
  m_cells[3].push_back(button);
}

void FWGUIEventFilter::newEntry(const char* text){
  if (m_haveNewEntry){
    // disconnect the last
    m_cells[0].back()->Disconnect("Toggled(bool)",&m_newEntry, "enable(bool)");
    m_cells[1].back()->Disconnect("TextChanged(char*)", &m_newEntry.selection, "assign(char*)");
    m_cells[1].back()->Disconnect("TextChanged(char*)", this, "newEntry(char*)");
    m_cells[2].back()->Disconnect("TextChanged(char*)", &m_newEntry.title, "assign(char*)");
    m_cells[3].back()->Disconnect("Clicked()", &m_newEntry, "remove()");
    m_cells[3].back()->Disconnect("Clicked()", this, "update()");
    m_sels.push_back(m_newEntry);
    m_cells[0].back()->Connect("Toggled(bool)","FWEventSelector", &m_sels.back(), "enable(bool)");
    m_cells[1].back()->Connect("TextChanged(char*)", "string",&m_sels.back().selection, "assign(char*)");
    m_cells[2].back()->Connect("TextChanged(char*)", "string",&m_sels.back().title, "assign(char*)");
    m_cells[3].back()->Connect("Clicked()","FWEventSelector", &m_sels.back(), "remove()");
    m_cells[3].back()->Connect("Clicked()","FWGUIEventFilter", this, "update()");
  }
  m_newEntry = FWEventSelector();
  addSelector(m_newEntry);
  m_cells[1].back()->Connect("TextChanged(char*)", "FWGUIEventFilter",this, "newEntry(char*)");
  m_cells[3].back()->Disconnect("Clicked()", this, "update()");
  m_cells[3].back()->Disconnect("Clicked()", &m_newEntry, "remove()");
  update();
  m_haveNewEntry = true;
}
void FWGUIEventFilter::show(){
  for(std::vector<FWEventSelector>::iterator sel = m_sels.begin();
      sel != m_sels.end(); ++sel)
    addSelector(*sel);
  update();
  newEntry();
}
 
void FWGUIEventFilter::update(){
  for(unsigned int i=0; i<m_sels.size(); ++i){
    if ( m_sels[i].removed && m_cells[0][i+1]!=0){
      m_columns[0]->RemoveFrame(m_cells[0][i+1]);
      m_columns[1]->RemoveFrame(m_cells[1][i+1]);
      m_columns[2]->RemoveFrame(m_cells[2][i+1]);
      m_columns[3]->RemoveFrame(m_cells[3][i+1]);
      m_cells[0][i+1]=0;
      m_cells[1][i+1]=0;
      m_cells[2][i+1]=0;
      m_cells[3][i+1]=0;
    }
  }
  MapSubwindows();
  Layout();
  MapWindow();
}
void FWGUIEventFilter::dump(const char* text){
  std::cout << "Text changed: " << text << std::endl;
  
  for(std::vector<FWEventSelector>::iterator sel = m_sels.begin();
      sel != m_sels.end(); ++sel)
    std::cout << "\t" << sel->enabled << "\t " << sel->selection << "\t" << sel->title<< 
      "\t " << sel->removed << std::endl;
}

void FWGUIEventFilter::junctionChanged(){
  m_globalOR = !m_globalOR;
  junctionUpdate();
}

void FWGUIEventFilter::junctionUpdate(){
  m_junctionWidget->SetDown(!m_globalOR);
  if (m_globalOR)
    m_junctionWidget->SetText(" OR  ");
  else
    m_junctionWidget->SetText(" AND ");
}
