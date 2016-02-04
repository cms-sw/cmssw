#include "EventFilter/Utilities/interface/TimeProfilerService.h"
#include "FWCore/Utilities/interface/Exception.h"
namespace evf{


  static double getTime()
  {
    struct timeval t;
    if(gettimeofday(&t,0)<0)
      throw cms::Exception("SysCallFailed","Failed call to gettimeofday");
    
    return (double)t.tv_sec + (double(t.tv_usec) * 1E-6);
  }
  
  TimeProfilerService::TimeProfilerService(const edm::ParameterSet& iPS, edm::ActivityRegistry&iRegistry)
  {
    iRegistry.watchPostBeginJob(this,&TimeProfilerService::postBeginJob);
    iRegistry.watchPostEndJob(this,&TimeProfilerService::postEndJob);
    
    iRegistry.watchPreProcessEvent(this,&TimeProfilerService::preEventProcessing);
    iRegistry.watchPostProcessEvent(this,&TimeProfilerService::postEventProcessing);
    
    iRegistry.watchPreModule(this,&TimeProfilerService::preModule);
    iRegistry.watchPostModule(this,&TimeProfilerService::postModule);
  }

  TimeProfilerService::~TimeProfilerService()
  {}
  
  void TimeProfilerService::postBeginJob()
  {}
  
  void TimeProfilerService::postEndJob()
  {}
  void TimeProfilerService::preEventProcessing(const edm::EventID& iID,
                                    const edm::Timestamp& iTime)
  {}
  void TimeProfilerService::postEventProcessing(const edm::Event& e, const edm::EventSetup&)
  {}
  void TimeProfilerService::preModule(const edm::ModuleDescription&)
  {
    curr_module_time_ = getTime();
  }

  void TimeProfilerService::postModule(const edm::ModuleDescription& desc)
  {
    double t = getTime() - curr_module_time_;
    std::map<std::string, times>::iterator it = profiles_.find(desc.moduleLabel());
    if(it==profiles_.end())
      {
	times tt;
	tt.ncalls_ = 0;
	tt.total_ = 0.;
	tt.max_ = 0.;
	tt.firstEvent_ = t;
	profiles_.insert(std::pair<std::string, times>(desc.moduleLabel(),tt));
      }      
    else
      {
	(*it).second.ncalls_++;
	(*it).second.total_ += t;
	(*it).second.max_  = ((*it).second.max_ > t) ? (*it).second.max_ : t;
      }
  }
  double TimeProfilerService::getFirst(std::string const &name) const
  {
    std::map<std::string, times>::const_iterator it = profiles_.find(name);

    if(it==profiles_.end())
      return -1.;
    return (*it).second.firstEvent_;
  }

  double TimeProfilerService::getMax(std::string const &name) const
  {
    std::map<std::string, times>::const_iterator it = profiles_.find(name);

    if(it == profiles_.end())
      return -1.;
    return (*it).second.max_;
  }

  double TimeProfilerService::getAve(std::string const &name) const
  {
    std::map<std::string, times>::const_iterator it = profiles_.find(name);

    if(it == profiles_.end())
      return -1.;
    return (*it).second.total_/(*it).second.ncalls_;
  }
}
