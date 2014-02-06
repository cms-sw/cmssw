// C++ headers
#include <string>
#include <cstring>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"

class FastTimerFilter : public edm::global::EDFilter<> {
public:
  explicit FastTimerFilter(edm::ParameterSet const &);
  ~FastTimerFilter();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  const double m_time_limit_event;
  const double m_time_limit_path;
  const double m_time_limit_allpaths;

  bool filter(edm::StreamID sid, edm::Event & event, const edm::EventSetup & setup) const override;
};

FastTimerFilter::FastTimerFilter(edm::ParameterSet const & config) :
  m_time_limit_event(    config.getParameter<double>( "timeLimitPerEvent" )),
  m_time_limit_path(     config.getParameter<double>( "timeLimitPerPath" )),
  m_time_limit_allpaths( config.getParameter<double>( "timeLimitPerAllPaths" ))
{
}

FastTimerFilter::~FastTimerFilter()
{
}

bool
FastTimerFilter::filter(edm::StreamID sid, edm::Event & event, edm::EventSetup const & setup) const
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return false;

  /* FIXME re-enable
  FastTimerService const & fts = * edm::Service<FastTimerService>();
  if (m_time_limit_allpaths > 0. and fts.queryPathsTime(sid)   > m_time_limit_allpaths)
    return true;
  if (m_time_limit_event    > 0. and fts.currentEventTime(sid) > m_time_limit_event)
    return true;
  if (m_time_limit_path     > 0. and fts.currentPathTime(sid)  > m_time_limit_path)
    return true;
  */

  return false;
}

void
FastTimerFilter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<double>("timeLimitPerEvent",      0.);
  desc.add<double>("timeLimitPerPath",       0.);
  desc.add<double>("timeLimitPerAllPaths", 120.);
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FastTimerFilter);
