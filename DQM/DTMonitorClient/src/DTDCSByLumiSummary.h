
#ifndef DTMonitorClient_DTDCSByLumiSummary_H
#define DTMonitorClient_DTDCSByLumiSummary_H

/** \class DTDCSByLumiSummary
 *  No description available.
 *
 *  \author C. Battilana - CIEMAT
 *  \author P. Bellan - INFN PD
 *  \author A. Branca = INFN PD
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <map>

class DTTimeEvolutionHisto;

class DTDCSByLumiSummary : public DQMEDHarvester {
public:
  /// Constructor
  DTDCSByLumiSummary(const edm::ParameterSet& pset);

  /// Destructor
  ~DTDCSByLumiSummary() override;

protected:
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                             DQMStore::IGetter& igetter,
                             edm::LuminosityBlock const& lumi,
                             edm::EventSetup const& setup) override;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;

private:
  MonitorElement* totalDCSFraction;
  MonitorElement* globalHVSummary;

  std::vector<DTTimeEvolutionHisto*> hDCSFracTrend;
  std::vector<MonitorElement*> totalDCSFractionWh;

  std::map<int, std::vector<float> > dcsFracPerLumi;

  bool bookingdone;
};

#endif
