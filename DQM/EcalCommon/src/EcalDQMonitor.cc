#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"

#include <sstream>

EcalDQMonitor::EcalDQMonitor(const edm::ParameterSet &_ps) :
  moduleName_(_ps.getUntrackedParameter<std::string>("moduleName", "Ecal Monitor")),
  mergeRuns_(_ps.getUntrackedParameter<bool>("mergeRuns", false)),
  verbosity_(_ps.getUntrackedParameter<int>("verbosity", 0)),
  initialized_(false)
{
}

EcalDQMonitor::~EcalDQMonitor()
{
}
