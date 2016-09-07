// -*- C++ -*-
//
// Package:     Services
// Class  :     CPU
// 
// Implementation:
//
// Original Author:  Natalia Garcia
// CPU.cc: v 1.0 2009/01/08 11:31:07


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <set>

#ifdef __linux__
#include <sched.h>
#include <errno.h>
#endif

namespace edm {
  
  namespace service {
    class CPU {
    public:
      CPU(ParameterSet const&, ActivityRegistry&);
      ~CPU();
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      
    private:
      int totalNumberCPUs_;
      double averageCoreSpeed_;
      bool reportCPUProperties_;
      
      void postEndJob();
    };
    
    inline
    bool isProcessWideService(CPU const*) {
      return true;
    }
  }
}

namespace edm {
  namespace service {
    namespace {

      std::string i2str(int i){
	std::ostringstream t;
	t << i;
	return t.str();
      }

      std::string d2str(double d){
	std::ostringstream t;
	t << d;
	return t.str();
      }

      double str2d(std::string s){
	return atof(s.c_str());
      }

      inline int str2i(std::string s){
	return atoi(s.c_str());
      }

      void trim(std::string& s, const std::string& drop = " \t") {
        std::string::size_type p = s.find_last_not_of(drop);
        if(p != std::string::npos) {
          s = s.erase(p+1);
        }
        s = s.erase(0, s.find_first_not_of(drop));
      }

      std::string eraseExtraSpaces(std::string s) {
	bool founded = false; 
	std::string aux;
        for(std::string::const_iterator iter = s.begin(); iter != s.end(); iter++){
		if(founded){
                        if(*iter == ' ') founded = true;
                        else{
                                aux += " "; aux += *iter;
				founded = false;
			}
		}
		else{
			if(*iter == ' ') founded = true;
			else aux += *iter;
		}
	}
	return aux;
      }

      // Determine the CPU set size; if this can be successfully determined, then this
      // returns true.
      bool getCpuSetSize(unsigned &set_size) {
#ifdef __linux__
        cpu_set_t *cpusetp;
        unsigned current_size = 128;
        unsigned cpu_count = 0;
        while (current_size*2 > current_size) {
          cpusetp = CPU_ALLOC(current_size);
          CPU_ZERO_S(CPU_ALLOC_SIZE(current_size), cpusetp);

          if (sched_getaffinity(0, CPU_ALLOC_SIZE(current_size), cpusetp)) {
            CPU_FREE(cpusetp);
            if (errno == EINVAL) {
              current_size *= 2;
              continue;
            }
            return false;
          }
          cpu_count = CPU_COUNT_S(CPU_ALLOC_SIZE(current_size), cpusetp);
          CPU_FREE(cpusetp);
          break;
        }
        set_size = cpu_count;
        return true;
#else
        return false;
#endif
      }
    } // namespace {}


    CPU::CPU(const ParameterSet& iPS, ActivityRegistry&iRegistry):
	totalNumberCPUs_(0),
	averageCoreSpeed_(0.0),
	reportCPUProperties_(iPS.getUntrackedParameter<bool>("reportCPUProperties"))
    {
	iRegistry.watchPostEndJob(this,&CPU::postEndJob);
    }


    CPU::~CPU()
    {
    }

    void CPU::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<bool>("reportCPUProperties", false);
      descriptions.add("CPU", desc);
    }


    void CPU::postEndJob()
    {
      Service<JobReport> reportSvc;

      std::map<std::string, std::string> reportCPUProperties; // Summary
      std::map<std::string, std::string> currentCoreProperties; // Module(s)

      std::ifstream fcpuinfo ("/proc/cpuinfo");

      if(fcpuinfo.is_open()){

	std::string buf;
	std::string currentCore;
	std::string CPUModels;

	std::set<std::string> models;

	while(!fcpuinfo.eof()){

		std::getline(fcpuinfo, buf);

	        std::istringstream iss(buf);
                std::string token;
                std::string property;
                std::string value;

                int time = 1;

                while(std::getline(iss, token, ':')) {
                        switch(time){
                        case 1:
                                property = token;
                                break;
                        case 2:
                                value = token;
                                break;
                        default:
                                value += token;
                                break;
                        }
                        time++;
                }
                trim(property);
                trim(value);

		if(!property.empty()){
			if(property == "processor") {
			    if(reportCPUProperties_){
				if(currentCore.empty()) { // first core
					currentCore = value;
				}
				else{
					reportSvc->reportPerformanceForModule("SystemCPU", "CPU-"+currentCore, currentCoreProperties);
					currentCoreProperties.clear();
					currentCore = value;
				}
			    }
			    totalNumberCPUs_++;
			}
			else {
				if(reportCPUProperties_){
					currentCoreProperties.insert(std::make_pair(property, value));
				}
				if(property == "cpu MHz"){
					averageCoreSpeed_ += str2d(value);
				}
				if(property == "model name"){
					models.insert(eraseExtraSpaces(value));
				}
			}
		}
	} //while

	fcpuinfo.close();

	if(!currentCore.empty() && reportCPUProperties_) {
		reportSvc->reportPerformanceForModule("SystemCPU", "CPU-"+currentCore, currentCoreProperties);
	}

	reportCPUProperties.insert(std::make_pair("totalCPUs", i2str(totalNumberCPUs_)));
	
	if(totalNumberCPUs_ == 0){
		averageCoreSpeed_ = 0.0;
	}
	else{
		averageCoreSpeed_ = averageCoreSpeed_/totalNumberCPUs_;
	}
	
	reportCPUProperties.insert(std::make_pair("averageCoreSpeed", d2str(averageCoreSpeed_)));

	int model = 0;
	for(std::set<std::string>::const_iterator iter = models.begin(); iter != models.end(); iter++){
		if(model == 0)
			CPUModels += *iter;
		else
			CPUModels += ", " + *iter;
		model++;
	}
	reportCPUProperties.insert(std::make_pair("CPUModels", CPUModels));

        unsigned set_size = -1;
        if (getCpuSetSize(set_size)) {
          reportCPUProperties.insert(std::make_pair("cpusetCount", i2str(set_size)));
        }

	reportSvc->reportPerformanceSummary("SystemCPU", reportCPUProperties);

      } //if
    } //postEndJob
  } //service
}  //edm


using edm::service::CPU;
DEFINE_FWK_SERVICE(CPU);


