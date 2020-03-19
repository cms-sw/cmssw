#ifndef Fireworks_Core_FWL1TriggerTableView_h
#define Fireworks_Core_FWL1TriggerTableView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWL1TriggerTableView
//
/**\class FWL1TriggerTableView FWL1TriggerTableView.h Fireworks/Core/interface/FWL1TriggerTableView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Tue Jan 25 16:02:19 CET 2011
//

#include "Fireworks/Core/interface/FWTriggerTableView.h"

class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class FWL1TriggerTableView : public FWTriggerTableView {
public:
  FWL1TriggerTableView(TEveWindowSlot*);
  ~FWL1TriggerTableView() override {}

protected:
  void fillTable(fwlite::Event* event) override;
};

#endif
