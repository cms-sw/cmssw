#ifndef DQMOFFLINE_L1TRIGGER_L1TDIFFHARVESTING_H
#define DQMOFFLINE_L1TRIGGER_L1TDIFFHARVESTING_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include <vector>

namespace dqmoffline {
namespace l1t {

class L1TDiffHarvesting: public DQMEDHarvester {

public:
  L1TDiffHarvesting(const edm::ParameterSet& ps);
  ~L1TDiffHarvesting() override;

protected:

  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;

private:
  class L1TDiffPlotHandler {
  public:
    L1TDiffPlotHandler(const edm::ParameterSet & ps, std::string plotName);
    L1TDiffPlotHandler(const L1TDiffPlotHandler &handler); // needed for vector collection

    void computeDiff(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

    std::string dir1_;
    std::string dir2_;
    std::string outputDir_;
    std::string plotName_;

    MonitorElement* h1_;
    MonitorElement* h2_;
    MonitorElement* h_diff_;
    MonitorElement::Kind histType1_, histType2_;

    void loadHistograms(DQMStore::IGetter &igetter);
    bool isValid() const;
    void bookDiff(DQMStore::IBooker &ibooker);

  };

  typedef std::vector<L1TDiffPlotHandler> L1TDiffPlotHandlers;

  L1TDiffPlotHandlers plotHandlers_;
};

} // l1t
} // dqmoffline

#endif
