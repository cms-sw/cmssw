// -*- C++ -*-
//
// Package:     Services
// Class  :     PathTimerService
// 
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id: PathTimerService.cc,v 1.14 2008/07/30 09:35:08 wittich Exp $
//

#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sstream>
#include <math.h>

using namespace std ;
namespace edm {
    namespace service {

        static double getTime()  {
            struct timeval t;
            if(gettimeofday(&t,0)<0)
                throw cms::Exception("SysCallFailed","Failed call to gettimeofday");
      
            return (double)t.tv_sec + (double(t.tv_usec) * 1E-6);
        }

        edm::CPUTimer* PathTimerService::_CPUtimer = 0;

        PathTimerService::PathTimerService(const ParameterSet& iPS, ActivityRegistry&iRegistry):
            total_event_count_(0),
            _perfInfo(new HLTPerformanceInfo)
        {
            iRegistry.watchPostBeginJob(this,&PathTimerService::postBeginJob);
            iRegistry.watchPostEndJob(this,&PathTimerService::postEndJob);
            
            iRegistry.watchPreProcessEvent(this,&PathTimerService::preEventProcessing);
            iRegistry.watchPostProcessEvent(this,&PathTimerService::postEventProcessing);
      
            iRegistry.watchPreModule(this,&PathTimerService::preModule);
            iRegistry.watchPostModule(this,&PathTimerService::postModule);

            iRegistry.watchPostProcessPath(this,&PathTimerService::postPathProcessing);
            
            _myPS=iPS;

            if (!_CPUtimer) _CPUtimer = new edm::CPUTimer();
        }


        PathTimerService::~PathTimerService() {
            if (_CPUtimer) {
                delete _CPUtimer ;
                _CPUtimer = 0 ;
            }
        }


        void PathTimerService::postBeginJob() {

            edm::Service<edm::service::TriggerNamesService> tns;
            std::vector<std::string> trigPaths= tns->getTrigPaths();
            for ( unsigned int i=0; i<trigPaths.size(); i++) {
                _pathMapping[i]=trigPaths[i];
                HLTPerformanceInfo::Path hltPath(trigPaths[i]);
                std::vector<unsigned int> loc ; 
                const std::vector<std::string> modules=tns->getTrigPathModules(trigPaths[i]);
                unsigned int mIdx = 0 ; 
                _perfInfo->addPath(hltPath);
                for ( unsigned int j=0; j<modules.size(); j++) {
                    _moduleTime[modules[j]]=0. ;
                    _moduleCPUTime[modules[j]]=0. ;
                    HLTPerformanceInfo::Modules::const_iterator iMod =
                        _perfInfo->findModule(modules[j].c_str());
                    if ( iMod == _perfInfo->endModules() ) {
                        HLTPerformanceInfo::Module hltModule(modules[j].c_str(),0.,0.,hlt::Ready);
                        _perfInfo->addModule(hltModule);
                    }

                    //--- Check the module frequency in the path ---//
                    bool duplicateModule = false ; 
                    for (unsigned int k=0; k<j; k++) {
                        if (modules[k] == modules[j]) {
                            if (!duplicateModule) loc.push_back(k) ; 
                            duplicateModule = true ;
                        }
                    }
                    if (!duplicateModule) {
		      _perfInfo->addModuleToPath(modules[j].c_str(),trigPaths[i].c_str());
                        loc.push_back(mIdx++) ; 
                    }
                }
                _newPathIndex.push_back(loc) ;
            }
            curr_job_ = getTime();
        }

    void PathTimerService::postEndJob() {}

    void PathTimerService::preEventProcessing(const edm::EventID& iID,
				    const edm::Timestamp& iTime) {
      curr_event_ = iID;
      curr_event_time_ = getTime();

      _perfInfo->clearModules();
      
      std::map<std::string, double>::iterator iter=_moduleTime.begin();
      std::map<std::string, double>::iterator iCPU=_moduleCPUTime.begin();

      while ( iter != _moduleTime.end()) {
	(*iter).second=0.;
	iter++;
      }
      while ( iCPU != _moduleCPUTime.end()) {
	(*iCPU).second=0.;
	iCPU++;
      }

    }
    void PathTimerService::postEventProcessing(const Event& e, const EventSetup&)  {
      total_event_count_ = total_event_count_ + 1;
    }

    void PathTimerService::preModule(const ModuleDescription&) {
        _CPUtimer->reset() ; 
        _CPUtimer->start() ; 
    }

      void PathTimerService::postModule(const ModuleDescription& desc) {

          _CPUtimer->stop() ;
          double tWall = _CPUtimer->realTime() ; 
          double tCPU  = _CPUtimer->cpuTime() ;
          _CPUtimer->reset() ; 
      
          _moduleTime[desc.moduleLabel()] = tWall ;
          _moduleCPUTime[desc.moduleLabel()] = tCPU ; 
        
          HLTPerformanceInfo::Modules::iterator iMod =
              _perfInfo->findModule(desc.moduleLabel().c_str());
          if ( iMod != _perfInfo->endModules() ) {
	    iMod->setTime(tWall) ;
	    iMod->setCPUTime(tCPU) ;
          }

      }

      void PathTimerService::postPathProcessing(const std::string &name, 
                                                const HLTPathStatus &status) {
          HLTPerformanceInfo::PathList::iterator iPath=_perfInfo->beginPaths();
          int ctr = 0 ; 
          while ( iPath != _perfInfo->endPaths() ) {
	    if ( iPath->name() == name) { 
	      unsigned int pIndex = _newPathIndex.at(ctr).at(status.index()) ;
	      iPath->setStatus(HLTPathStatus(status.state(),pIndex)) ; 
	      _perfInfo->setStatusOfModulesFromPath(name.c_str());
	    }
	    iPath++;
	    ctr++; 

	  }

      }
  }
}
