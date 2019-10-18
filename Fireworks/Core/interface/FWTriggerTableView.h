// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWStringParameter.h"

#include <boost/unordered_map.hpp>

// forward declarations
class TGFrame;
class TGCompositeFrame;
class FWTableWidget;
class TGComboBox;
class TEveWindowFrame;
class TEveWindowSlot;
class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class FWJobMetadataManager;

namespace fwlite {
  class Event;
}

class FWTriggerTableView : public FWViewBase {
  friend class FWTriggerTableViewTableManager;

public:
  struct Column {
    std::string title;
    std::vector<std::string> values;
    Column(const std::string& s) : title(s) {}
  };

  FWTriggerTableView(TEveWindowSlot*, FWViewType::EType);
  ~FWTriggerTableView(void) override;

  // ---------- const member functions ---------------------
  void addTo(FWConfiguration&) const override;
  void saveImageTo(const std::string& iName) const override;
  Color_t backgroundColor() const { return m_backgroundColor; }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setFrom(const FWConfiguration&) override;
  void setBackgroundColor(Color_t);
  //void resetColors( const class FWColorManager& );
  void dataChanged(void);
  void columnSelected(Int_t iCol, Int_t iButton, Int_t iKeyMod);

  void setProcessList(std::vector<std::string>* x) { m_processList = x; }
  void resetCombo() const;
  //  void processChanged(Int_t);
  void processChanged(const char*);

protected:
  FWStringParameter m_regex;
  FWStringParameter m_process;

  std::vector<Column> m_columns;
  FWTriggerTableViewTableManager* m_tableManager;

  virtual void fillTable(fwlite::Event* event) = 0;

private:
  FWTriggerTableView(const FWTriggerTableView&) = delete;                   // stop default
  const FWTriggerTableView& operator=(const FWTriggerTableView&) = delete;  // stop default

  bool isProcessValid() const;
  void populateController(ViewerParameterGUI&) const override;

  mutable TGComboBox* m_combo;

  // destruction
  TEveWindowFrame* m_eveWindow;
  TGCompositeFrame* m_vert;

  FWTableWidget* m_tableWidget;

  Color_t m_backgroundColor;

  std::vector<std::string>* m_processList;
};

#endif
