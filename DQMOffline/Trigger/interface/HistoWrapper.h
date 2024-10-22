#ifndef __HistoWrapper__
#define __HistoWrapper__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
typedef dqm::legacy::DQMStore DQMStore;
#include "DQMServices/Core/interface/MonitorElement.h"
typedef dqm::legacy::MonitorElement MonitorElement;

enum PL { kEverything, kVital };

class HistoWrapper {
public:
  HistoWrapper(const edm::ParameterSet&);
  ~HistoWrapper();

  MonitorElement* book1D(DQMStore::IBooker& iBooker,
                         TString const& name,
                         TString const& title,
                         int const nchX,
                         double const lowX,
                         double const highX,
                         int level = kEverything);
  MonitorElement* book2D(DQMStore::IBooker& iBooker,
                         TString const& name,
                         TString const& title,
                         int nchX,
                         double lowX,
                         double highX,
                         int nchY,
                         double lowY,
                         double highY,
                         int level = kEverything);

private:
  PL plotlevel;
  int cAllHistograms;
  int cPlottedHistograms;
};
#endif
