// -*- C++ -*-
//
// Package:     Services
// Class  :     PrescaleServices
// 
// Implementation:
//     Cache and make prescale factors available online.
//
// Current revision: $Revision$
// On branch: $Name$
// Latest change by $Author$ on $Date$ 
//

#include "FWCore/Services/src/PrescaleService.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

namespace edm {
  namespace service {

    PrescaleService::PrescaleService(const ParameterSet& iPS, ActivityRegistry&iReg)
    {
      
      LogDebug("PrescaleService") << "PrescaleService::PrescaleService";

      iReg.watchPostBeginJob(this,&PrescaleService::postBeginJob);
      iReg.watchPostEndJob(this,&PrescaleService::postEndJob);
      
      iReg.watchPreProcessEvent(this,&PrescaleService::preEventProcessing);
      iReg.watchPostProcessEvent(this,&PrescaleService::postEventProcessing);

      iReg.watchPreModule(this,&PrescaleService::preModule);
      iReg.watchPostModule(this,&PrescaleService::postModule);

      count_ = 0;
      fu_ = 0;
      lsnr_ = -1;
    }

    PrescaleService::~PrescaleService()
    {
    }

    void PrescaleService::postBeginJob()
    {
    }

    void PrescaleService::postEndJob()
    {
    }

    void PrescaleService::preEventProcessing(const edm::EventID& iID,
					       const edm::Timestamp& iTime)
    {
      //      std::cout << "!!! PrescaleService::preEventProcessing count " << count_ << std::endl;
      if (count_%100 == 0) {
	if (fu_ != 0) {
	  //	  std::cout << "!!! get trigger report" << std::endl;
	  fu_->getTriggerReport(tr_);
	  lsnr_ += 1;
	  //	  std::cout << "!!! trigger report done" << std::endl;

	  ostringstream oss;
	  string ARRAY_LEN = "_";
	  string SEPARATOR = " ";
	  // Lumi section number
	  oss << lsnr_ << ":";
	  //TriggerReport::eventSummary
	  oss << tr_.eventSummary.totalEvents << SEPARATOR
	      << tr_.eventSummary.totalEventsPassed << SEPARATOR
	      << tr_.eventSummary.totalEventsFailed << SEPARATOR;
	  //TriggerReport::trigPathSummaries
	  oss << ARRAY_LEN << tr_.trigPathSummaries.size() << SEPARATOR;
	  for(unsigned int i=0; i<tr_.trigPathSummaries.size(); i++)
	    {
	      oss << tr_.trigPathSummaries[i].bitPosition << SEPARATOR
		  << tr_.trigPathSummaries[i].timesRun << SEPARATOR
		  << tr_.trigPathSummaries[i].timesPassed << SEPARATOR
		  << tr_.trigPathSummaries[i].timesFailed << SEPARATOR
		  << tr_.trigPathSummaries[i].timesExcept << SEPARATOR
		  << tr_.trigPathSummaries[i].name << SEPARATOR;
	      //TriggerReport::trigPathSummaries::moduleInPathSummaries
	      oss << ARRAY_LEN << tr_.trigPathSummaries[i].moduleInPathSummaries.size() << SEPARATOR;
	      for(unsigned int j=0; j<tr_.trigPathSummaries[i].moduleInPathSummaries.size(); j++)
		{
		  oss << tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited << SEPARATOR
		      << tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed << SEPARATOR
		      << tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed << SEPARATOR
		      << tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept << SEPARATOR
		      << tr_.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel << SEPARATOR;
		}
	    }

	  //TriggerReport::endPathSummaries
	  oss << ARRAY_LEN << tr_.endPathSummaries.size() << SEPARATOR;
	  for(unsigned int i=0; i<tr_.endPathSummaries.size(); i++)
	    {
	      oss << tr_.endPathSummaries[i].bitPosition << SEPARATOR
		  << tr_.endPathSummaries[i].timesRun << SEPARATOR
		  << tr_.endPathSummaries[i].timesPassed << SEPARATOR
		  << tr_.endPathSummaries[i].timesFailed << SEPARATOR
		  << tr_.endPathSummaries[i].timesExcept << SEPARATOR
		  << tr_.endPathSummaries[i].name << SEPARATOR;
	      //TriggerReport::endPathSummaries::moduleInPathSummaries
	      oss << ARRAY_LEN << tr_.endPathSummaries[i].moduleInPathSummaries.size() << SEPARATOR;
	      for(unsigned int j=0; j<tr_.endPathSummaries[i].moduleInPathSummaries.size(); j++)
		{
		  oss << tr_.endPathSummaries[i].moduleInPathSummaries[j].timesVisited << SEPARATOR
		      << tr_.endPathSummaries[i].moduleInPathSummaries[j].timesPassed << SEPARATOR
		      << tr_.endPathSummaries[i].moduleInPathSummaries[j].timesFailed << SEPARATOR
		      << tr_.endPathSummaries[i].moduleInPathSummaries[j].timesExcept << SEPARATOR
		      << tr_.endPathSummaries[i].moduleInPathSummaries[j].moduleLabel << SEPARATOR;
		}
	    }

	  //TriggerReport::workerSummaries
	  oss << ARRAY_LEN << tr_.workerSummaries.size() << SEPARATOR;
	  for(unsigned int i=0; i<tr_.workerSummaries.size(); i++)
	    {
	      oss << tr_.workerSummaries[i].timesVisited << SEPARATOR
		  << tr_.workerSummaries[i].timesRun     << SEPARATOR
		  << tr_.workerSummaries[i].timesPassed  << SEPARATOR
		  << tr_.workerSummaries[i].timesFailed  << SEPARATOR
		  << tr_.workerSummaries[i].timesExcept  << SEPARATOR
		  << tr_.workerSummaries[i].moduleLabel  << SEPARATOR;
	    }

	  trs_ = oss.str();
	  //	  std::cout << "!!! trigger string " << trs_ << std::endl;
	}
      }
    }

    void PrescaleService::postEventProcessing(const Event& e,
						const EventSetup&)
    {
	++count_;
    }

    void PrescaleService::preModule(const ModuleDescription& md)
    {
    }

    void PrescaleService::postModule(const ModuleDescription& md)
    {
    }

    unsigned int PrescaleService::getPrescale(unsigned int ls, string module)
    {
      return store.get(ls, module);
    }

    int PrescaleService::putPrescale(string s)
    {
      store.add(s);
      return store.size();
    }

    int PrescaleService::sizePrescale()
    {
      return store.size();
    }

    void PrescaleService::putHandle(edm::EventProcessor *proc_)
    {
      fu_ = proc_;
    }

    string PrescaleService::getTriggerCounters()
    {
      //      std::cout << "PrescaleService::getTriggerCounters lsnr_ " << lsnr_ << std::endl;
      return trs_;
    }
  }
}
