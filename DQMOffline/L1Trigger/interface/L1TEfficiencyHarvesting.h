#ifndef DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYHARVESTING_H
#define DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYHARVESTING_H

/**
 * \file L1TEfficiencyHarvesting.h
 *
 * \author J. Pela, C. Battilana
 *
 */

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <vector>

namespace dqmoffline {
namespace l1t {

//
// Efficiency helper class declaration
//

class L1TEfficiencyPlotHandler {

public:

  L1TEfficiencyPlotHandler(const edm::ParameterSet & ps, std::string plotName);

  L1TEfficiencyPlotHandler(const L1TEfficiencyPlotHandler &handler);

  ~L1TEfficiencyPlotHandler()
  {
  }
  ;

  // book efficiency histo
  void book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

  // compute efficiency
  void computeEfficiency(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

private:

  std::string numeratorDir_;
  std::string denominatorDir_;
  std::string outputDir_;
  std::string plotName_;
  std::string numeratorSuffix_;
  std::string denominatorSuffix_;

  MonitorElement* h_efficiency_;
};

typedef std::vector<L1TEfficiencyPlotHandler> L1TEfficiencyPlotHandlerCollection;

//
// DQM class declaration
//

class L1TEfficiencyHarvesting: public DQMEDHarvester {

public:

  L1TEfficiencyHarvesting(const edm::ParameterSet& ps);   // Constructor
  ~L1TEfficiencyHarvesting() override;                     // Destructor

protected:

  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  virtual void dqmEndLuminosityBlock(DQMStore::IGetter &igetter, edm::LuminosityBlock const& lumiBlock,
      edm::EventSetup const& c);

private:

  bool verbose_;

  L1TEfficiencyPlotHandlerCollection plotHandlers_;

};

} //l1t
} // dqmoffline

#endif
