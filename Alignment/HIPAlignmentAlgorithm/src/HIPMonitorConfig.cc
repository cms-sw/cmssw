#include <Alignment/HIPAlignmentAlgorithm/interface/HIPMonitorConfig.h>


HIPMonitorConfig::HIPMonitorConfig(const edm::ParameterSet& cfg) :
cfgMonitor(cfg.getParameter<edm::ParameterSet>("monitorConfig")),
outfilecore(cfgMonitor.getParameter<std::string>("outfile")),
fillTrackMonitoring(cfgMonitor.getParameter<bool>("fillTrackMonitoring")),
maxEventsPerJob(cfgMonitor.getParameter<int>("maxEventsPerJob")),
fillTrackHitMonitoring(cfgMonitor.getParameter<bool>("fillTrackHitMonitoring")),
maxHits(cfgMonitor.getParameter<int>("maxHits")),
eventCounter(0),
hitCounter(0)
{
  outfile = cfg.getParameter<std::string>("outpath") + outfilecore;
}

HIPMonitorConfig::HIPMonitorConfig(const HIPMonitorConfig& other) :
cfgMonitor(other.cfgMonitor),
outfilecore(other.outfilecore),
fillTrackMonitoring(other.fillTrackMonitoring),
maxEventsPerJob(other.maxEventsPerJob),
fillTrackHitMonitoring(other.fillTrackHitMonitoring),
maxHits(other.maxHits),
outfile(other.outfile),
eventCounter(other.eventCounter),
hitCounter(other.hitCounter)
{}

bool HIPMonitorConfig::checkNevents(){ bool res = (maxEventsPerJob<0 || maxEventsPerJob>eventCounter); eventCounter++; return res; }
bool HIPMonitorConfig::checkNhits(){ bool res = (maxHits<0 || maxHits>hitCounter); hitCounter++; return res; }

