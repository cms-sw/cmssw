#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

ESClient::ESClient(edm::ParameterSet const& _ps) :
  initialized_(false),
  prefixME_(_ps.getUntrackedParameter<std::string>("prefixME")),
  cloneME_(_ps.getUntrackedParameter<bool>("cloneME")),
  verbose_(_ps.getUntrackedParameter<bool>("verbose")),
  debug_(_ps.getUntrackedParameter<bool>("debug"))
{
}

void
ESClient::setup(DQMStore::IBooker& _ibooker)
{
  if(initialized_) return;
  book(_ibooker);
  initialized_ = true;
}
