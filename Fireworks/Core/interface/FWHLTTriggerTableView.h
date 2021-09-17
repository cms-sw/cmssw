#ifndef Fireworks_Core_FWHLTTriggerTableView_h
#define Fireworks_Core_FWHLTTriggerTableView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTTriggerTableView
//
/**\class FWHLTTriggerTableView FWHLTTriggerTableView.h Fireworks/Core/interface/FWHLTTriggerTableView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Tue Jan 25 16:02:24 CET 2011
//

#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWStringParameter.h"

#include <unordered_map>

class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class ViewerParameterGUI;

class FWHLTTriggerTableView : public FWTriggerTableView {
public:
  FWHLTTriggerTableView(TEveWindowSlot*);
  ~FWHLTTriggerTableView() override {}

protected:
  void fillTable(fwlite::Event* event) override;

private:
  typedef std::unordered_map<std::string, double> acceptmap_t;

  fwlite::Event* m_event;
  acceptmap_t m_averageAccept;

  void fillAverageAcceptFractions();
};
#endif
