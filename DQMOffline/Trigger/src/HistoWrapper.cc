#include "DQMOffline/Trigger/interface/HistoWrapper.h"

HistoWrapper::HistoWrapper(const edm::ParameterSet& pset) {
  plotlevel = (PL)pset.getUntrackedParameter<int>("PlotLevel", int(kEverything));
  cAllHistograms = 0;
  cPlottedHistograms = 0;
}

HistoWrapper::~HistoWrapper() {
  std::string s_pl = "kEverything";
  if (plotlevel == 1)
    s_pl = "kVital";
  std::cout << "Plot level " << plotlevel << " " << s_pl << std::endl;
  std::cout << "Plotting " << cPlottedHistograms << " out of " << cAllHistograms << std::endl;
}

MonitorElement* HistoWrapper::book1D(DQMStore::IBooker& iBooker,
                                     TString const& name,
                                     TString const& title,
                                     int const nchX,
                                     double const lowX,
                                     double const highX,
                                     int level) {
  cAllHistograms++;
  if (level >= plotlevel) {
    cPlottedHistograms++;
    MonitorElement* me = iBooker.book1D(name, title, nchX, lowX, highX);
    return me;
  }
  return nullptr;
}

MonitorElement* HistoWrapper::book2D(DQMStore::IBooker& iBooker,
                                     TString const& name,
                                     TString const& title,
                                     int nchX,
                                     double lowX,
                                     double highX,
                                     int nchY,
                                     double lowY,
                                     double highY,
                                     int level) {
  cAllHistograms++;
  if (level >= plotlevel) {
    cPlottedHistograms++;
    MonitorElement* me = iBooker.book2D(name, title, nchX, lowX, highX, nchY, lowY, highY);
    return me;
  }
  return nullptr;
}
