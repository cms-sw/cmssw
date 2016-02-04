#ifndef EvFTimeProfilerService_H
#define EvFTimeProfilerService_H 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "boost/thread/thread.hpp"
#include <sys/time.h>

#include <string>
#include <vector>
#include <map>

namespace evf {

    class TimeProfilerService
    {
    public:
      TimeProfilerService(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~TimeProfilerService();

      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preSource();
      void postSource();
      
      
      void preModule(const edm::ModuleDescription&);
      void postModule(const edm::ModuleDescription&);
      double getFirst(std::string const &name) const; 
      double getMax(std::string const &name) const;
      double getAve(std::string const &name) const;

    private:

      boost::mutex lock_;
      double curr_module_time_; // seconds
      struct times{
	double  firstEvent_;
	double  max_;
	double  total_;
	int     ncalls_;
      };
      std::map<std::string, times> profiles_;
    };

}

#endif
