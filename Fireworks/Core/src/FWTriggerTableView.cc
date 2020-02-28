// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTriggerTableView
//

// system include files
#include <fstream>
#include <cassert>

#include "boost/bind.hpp"

#include "TEveWindow.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
//#include "TGHorizontalFrame.h"

// user include files
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWTriggerTableViewTableManager.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "DataFormats/FWLite/interface/Event.h"

// configuration keys
static const std::string kColumns = "columns";
static const std::string kSortColumn = "sortColumn";
static const std::string kDescendingSort = "descendingSort";

//
//
// constructors and destructor
//
FWTriggerTableView::FWTriggerTableView(TEveWindowSlot* iParent, FWViewType::EType id)
    : FWViewBase(id, 2),
      m_regex(this, "Filter", std::string()),
      m_process(this, "Process", std::string((id == FWViewType::FWViewType::kTableHLT) ? "HLT" : "")),
      m_tableManager(new FWTriggerTableViewTableManager(this)),
      m_combo(nullptr),
      m_eveWindow(nullptr),
      m_vert(nullptr),
      m_tableWidget(nullptr),
      m_processList(nullptr) {
  m_regex.changed_.connect(boost::bind(&FWTriggerTableView::dataChanged, this));

  m_eveWindow = iParent->MakeFrame(nullptr);
  TGCompositeFrame* frame = m_eveWindow->GetGUICompositeFrame();

  m_vert = new TGVerticalFrame(frame);
  frame->AddFrame(m_vert, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

  // have to have at least one column when call  FWTableWidget constructor
  m_columns.push_back(Column("Name"));

  m_tableWidget = new FWTableWidget(m_tableManager, frame);
  m_tableWidget->SetHeaderBackgroundColor(gVirtualX->GetPixel(kWhite));

  m_tableWidget->Connect(
      "columnClicked(Int_t,Int_t,Int_t)", "FWTriggerTableView", this, "columnSelected(Int_t,Int_t,Int_t)");
  m_vert->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

  frame->MapSubwindows();
  frame->Layout();
  frame->MapWindow();
}

FWTriggerTableView::~FWTriggerTableView() {
  // take out composite frame and delete it directly (without the timeout)
  TGCompositeFrame* frame = m_eveWindow->GetGUICompositeFrame();
  frame->RemoveFrame(m_vert);
  delete m_vert;

  m_eveWindow->DestroyWindowAndSlot();
  delete m_tableManager;
}

void FWTriggerTableView::setBackgroundColor(Color_t iColor) {
  m_backgroundColor = iColor;
  m_tableWidget->SetBackgroundColor(gVirtualX->GetPixel(iColor));
  m_tableWidget->SetLineSeparatorColor(gVirtualX->GetPixel(iColor == kWhite ? kBlack : kWhite));
}

//
// const member functions
//

/*
void
FWTriggerTableView::saveImageTo( const std::string& iName ) const
{
   std::string fileName = iName + ".txt";
   std::ofstream triggers( fileName.c_str() );

   triggers << m_columns[2].title << " " << m_columns[0].title << "\n";
   for( unsigned int i = 0, vend = m_columns[0].values.size(); i != vend; ++i )
      if( m_columns[1].values[i] == "1" )
         triggers << m_columns[2].values[i] << "\t" << m_columns[0].values[i] << "\n";
   triggers.close();
   }*/

void FWTriggerTableView::dataChanged() {
  for (std::vector<Column>::iterator i = m_columns.begin(); i != m_columns.end(); ++i)
    (*i).values.clear();

  edm::EventBase* base = const_cast<edm::EventBase*>(FWGUIManager::getGUIManager()->getCurrentEvent());
  if (fwlite::Event* event = dynamic_cast<fwlite::Event*>(base))
    fillTable(event);

  m_tableManager->dataChanged();
}

void FWTriggerTableView::columnSelected(Int_t iCol, Int_t iButton, Int_t iKeyMod) {}

//______________________________________________________________________________

void FWTriggerTableView::addTo(FWConfiguration& iTo) const {
  FWConfigurableParameterizable::addTo(iTo);
  FWConfiguration sortColumn(m_tableWidget->sortedColumn());
  iTo.addKeyValue(kSortColumn, sortColumn);
  FWConfiguration descendingSort(m_tableWidget->descendingSort());
  iTo.addKeyValue(kDescendingSort, descendingSort);
}

void FWTriggerTableView::setFrom(const FWConfiguration& iFrom) {
  const FWConfiguration* main = &iFrom;

  // unnecessary nesting for old version
  if (version() < 2) {
    if (typeId() == FWViewType::kTableHLT)
      main = iFrom.valueForKey("HLTTriggerTableView");
    else
      main = iFrom.valueForKey("L1TriggerTableView");
  }

  const FWConfiguration* sortColumn = main->valueForKey(kSortColumn);
  const FWConfiguration* descendingSort = main->valueForKey(kDescendingSort);
  if (sortColumn != nullptr && descendingSort != nullptr) {
    unsigned int sort = sortColumn->version();
    bool descending = descendingSort->version();
    if (sort < ((unsigned int)m_tableManager->numberOfColumns()))
      m_tableWidget->sort(sort, descending);
  }

  if (typeId() == FWViewType::kTableHLT) {
    const FWConfiguration* vp = iFrom.valueForKey("Process");
    if (vp && (vp->value() != m_process.value()))
      m_process.setFrom(iFrom);
  }

  {
    const FWConfiguration* vp = iFrom.valueForKey("Filter");
    if (vp && (vp->value() != m_regex.value()))
      m_regex.setFrom(iFrom);
  }
}

//______________________________________________________________________________

void FWTriggerTableView::resetCombo() const {
  if (m_combo && m_processList) {
    m_combo->RemoveAll();
    int cnt = 0;
    int id = -1;
    for (std::vector<std::string>::iterator i = m_processList->begin(); i != m_processList->end(); ++i) {
      if (m_process.value() == *i)
        id = cnt;

      m_combo->AddEntry((*i).c_str(), cnt);
      cnt++;
    }

    if (id < 0) {
      // fwLog(fwlog::kWarning) << "FWTriggerTableView: no trigger results with process name "<< m_process.value() << " is available" << std::endl;
      m_combo->AddEntry(m_process.value().c_str(), cnt);
      id = cnt;
    }

    m_combo->SortByName();
    m_combo->Select(id, false);
  }
}

void FWTriggerTableView::processChanged(const char* x) {
  m_process.set(x);
  dataChanged();
}

bool FWTriggerTableView::isProcessValid() const {
  for (std::vector<std::string>::iterator i = m_processList->begin(); i != m_processList->end(); ++i) {
    if (*i == m_process.value())
      return true;
  }
  return false;
}

void FWTriggerTableView::populateController(ViewerParameterGUI& gui) const {
  gui.requestTab("Style").addParam(&m_regex);

  // resize filter frame
  TGCompositeFrame* parent = gui.getTabContainer();
  TGFrameElement* el = (TGFrameElement*)parent->GetList()->Last();
  el->fLayout->SetLayoutHints(kLHintsNormal);
  el->fFrame->Resize(180);

  // add combo for processes
  if (typeId() == FWViewType::kTableHLT) {
    TGHorizontalFrame* f = new TGHorizontalFrame(gui.getTabContainer());
    gui.getTabContainer()->AddFrame(f, new TGLayoutHints(kLHintsNormal, 2, 2, 2, 2));

    m_combo = new TGComboBox(f);
    f->AddFrame(m_combo);
    m_combo->Resize(140, 20);
    f->AddFrame(new TGLabel(f, "Process"), new TGLayoutHints(kLHintsLeft, 8, 2, 2, 2));

    resetCombo();
    FWTriggerTableView* tt = (FWTriggerTableView*)this;
    m_combo->Connect("Selected(const char*)", "FWTriggerTableView", tt, "processChanged(const char*)");
  }
}

void FWTriggerTableView::saveImageTo(const std::string& /*iName*/) const {
  TString format;
  TString data;
  FWTextTableCellRenderer* textRenderer;

  // calculate widths
  std::vector<size_t> widths(m_tableManager->numberOfColumns());

  for (int c = 0; c < m_tableManager->numberOfColumns(); ++c)
    widths[c] = m_columns[c].title.size();

  for (int c = 0; c < m_tableManager->numberOfColumns(); ++c) {
    for (int r = 0; r < m_tableManager->numberOfRows(); r++) {
      textRenderer = (FWTextTableCellRenderer*)m_tableManager->cellRenderer(r, c);  // setup cell renderer
      size_t ss = textRenderer->data().size();
      if (widths[c] < ss)
        widths[c] = ss;
    }
  }

  int rlen = 0;
  for (size_t c = 0; c < (size_t)m_tableManager->numberOfColumns(); ++c)
    rlen += widths[c];
  rlen += (m_tableManager->numberOfColumns() - 1) * 3;
  rlen++;

  printf("\n");
  int lastCol = m_tableManager->numberOfColumns() - 1;

  for (int c = 0; c < m_tableManager->numberOfColumns(); ++c) {
    format.Form("%%%ds", (int)widths[c]);
    data.Form(format, m_columns[c].title.c_str());
    if (c == lastCol)
      printf("%s", data.Data());
    else
      printf("%s | ", data.Data());
  }
  printf("\n");

  std::string splitter(rlen, '-');
  std::cout << splitter << std::endl;

  for (int r = 0; r < m_tableManager->numberOfRows(); r++) {
    for (int c = 0; c < m_tableManager->numberOfColumns(); ++c) {
      format.Form("%%%ds", (int)widths[c]);
      textRenderer = (FWTextTableCellRenderer*)m_tableManager->cellRenderer(r, c);  // setup cell renderer
      data.Form(format, textRenderer->data().c_str());
      if (c == lastCol)
        printf("%s", data.Data());
      else
        printf("%s | ", data.Data());
    }
    printf("\n");
  }
}
