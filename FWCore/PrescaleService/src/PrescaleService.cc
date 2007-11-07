// -*- C++ -*-
//
// Package:     Services
// Class  :     PrescaleServices
// 
// Implementation:
//     Cache and make prescale factors available online.
//
// Current revision: $Revision: 1.9 $
// On branch: $Name: CMSSW_1_7_0_pre7 $
// Latest change by $Author: wmtan $ on $Date: 2007/09/17 16:06:24 $ 
//


#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>
#include <algorithm>

namespace edm {
  namespace service {

    PrescaleService::PrescaleService(const ParameterSet& iPS, ActivityRegistry&iReg)
    {
      blsn = bpath = bmod = bfac = berr = 0;
      lsgmax = glow = 0;
      lspmax = pleq = 0;
      lsg = lsc = bang = 0;
      count_ = 0;
      fu_ = 0;
      lsold = 0;
      nops = 0; 
      bcfg = 0;

      //
      const std::vector<std::string> InitialConfig(iPS.getParameter< std::vector<std::string> >("InitialConfig"));
      for (unsigned int I=0; I!=InitialConfig.size(); ++I) {
	const int i(putPrescale(InitialConfig[I]));
	if (i-1!=static_cast<int>(I)) LogDebug("PrescaleService")
	  << "Invalid config string " << I << ": '"
	  << InitialConfig[I] << "' - Ignored!";
      }
      //

      LogDebug("PrescaleService") << "PrescaleService::PrescaleService: "
				  << prescalers.size() << " of "
				  << InitialConfig.size() << " initialised!";

      iReg.watchPostBeginJob(this,&PrescaleService::postBeginJob);
      iReg.watchPostEndJob(this,&PrescaleService::postEndJob);

      iReg.watchPreProcessEvent(this,&PrescaleService::preEventProcessing);
      iReg.watchPostProcessEvent(this,&PrescaleService::postEventProcessing);

      iReg.watchPreModule(this,&PrescaleService::preModule);
      iReg.watchPostModule(this,&PrescaleService::postModule);

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
      if (fu_ != 0) {
//        std::cout << "!!! get trigger report" << std::endl;
        fu_->getTriggerReport(trold);
      }
    }

    void PrescaleService::postEventProcessing(const edm::Event& e, const edm::EventSetup& c)
    {
      if (fu_ != 0) {
        if ((count_ != 0)&&(e.luminosityBlock() != lsold)) {
//        if ((count_ != 0)&&(count_/100 != lsold)) {  //test//
	  std::ostringstream oss;
	  std::string ARRAY_LEN = "_";
	  std::string SEPARATOR = " ";
	  oss << lsold << SEPARATOR;
	  //TriggerReport::eventSummary
	  oss << trold.eventSummary.totalEvents << SEPARATOR
	      << trold.eventSummary.totalEventsPassed << SEPARATOR
	      << trold.eventSummary.totalEventsFailed << SEPARATOR;
	  //TriggerReport::trigPathSummaries
	  oss << ARRAY_LEN << trold.trigPathSummaries.size() << SEPARATOR;
	  for(unsigned int i=0; i<trold.trigPathSummaries.size(); i++) {
            oss << trold.trigPathSummaries[i].bitPosition << SEPARATOR
		<< trold.trigPathSummaries[i].timesRun << SEPARATOR
		<< trold.trigPathSummaries[i].timesPassed << SEPARATOR
		<< trold.trigPathSummaries[i].timesFailed << SEPARATOR
		<< trold.trigPathSummaries[i].timesExcept << SEPARATOR
		<< trold.trigPathSummaries[i].name << SEPARATOR;
	  }
          boost::mutex::scoped_lock scoped_lock(mutex);
	  triggers.push_back(oss.str());
	}
	lsold = e.luminosityBlock();
//	lsold = count_/100;          //test//
      }

//        edm::Timestamp t = e.time();
//        std::cout << "Event time " << t.value() << std::endl;
//        std::cout << "Run# " << e.run() << std::endl;
//        std::cout << "Event LS# " << e.luminosityBlock() << std::endl; 

      ++count_;
    }

    void PrescaleService::preModule(const ModuleDescription& md)
    {
    }

    void PrescaleService::postModule(const ModuleDescription& md)
    {
    }

    // Prepare indexed access without LS#
    int PrescaleService::getPrescale(std::string module)
    {
      return getPrescale(0, module);
    }

    int PrescaleService::getPrescale(unsigned int ls, std::string module)
    {
      boost::mutex::scoped_lock scoped_lock(mutex);

      if (ls < lsgmax) {
	glow++;
      } else {
	lsgmax = ls;
      }

      if (prescalers.size()<=0) {
        nops++;
        return -1;
      }

      int j = prescalers.size()-1;
      for( ; j>=0; j--) {
//	cout << "getPrescale j " << j << std::endl;
	unsigned int n; 
	std::istringstream iss(prescalers[j]);
	iss >> n;
//	std::cout << "getPrescale n " << n << " ls " << ls << std::endl;
	if (ls >= n) {
//	    std::cout << "getPrescale n  " << n << " <= " << " ls " << ls << std::endl;
	    break;
	}
      }
      if (j < 0) j = 0;

      unsigned int i = j;
      unsigned int n, m;
      std::string a, b;
      std::istringstream iss(prescalers[i]);
      iss >> n;
      while ( iss.rdstate()==0 ) {
	iss >> a >> b >> m; 
//	std::cout << "getPrescale " << n << "==" << ls << " a " << a << " b " << b << " m " << m << std::endl;
	if ( (b=="*") || (b==module) ) { // allow wildcard after an explicit module list
            if (lsg == ls && lsc < n) {
		bang++;
		return -1;
	    }
	    lsg = ls;
	    lsc = n;
	    return (int)m;
	}  
      }

      return -1;
    }

    int PrescaleService::putPrescale(std::string s)
    {

      std::istringstream iss(s);
      unsigned int i, n, m;
      std::string a, b;
      
      iss >> n;
      if (iss.fail()||iss.eof()) {
	  blsn++;
	  return -1;
      }

      if ((n <= lspmax)&&(lspmax != 0)) {
	  pleq++;
	  return -1;
      }
      lspmax = n;

      for(i=0; iss.good(); i++) {
	iss >> a;
	if (iss.fail()||iss.eof()) {
	  if (i>0) break; // trailing space
	  bpath++;
	  return -1;
	}
	iss >> b;
	if (iss.fail()||iss.eof()) {
	  bmod++;
	  return -1;
	}
	iss >> m;
	if (iss.fail()) {
	  bfac++;
	  return -1;
	}
      }

      // validate
      if (i < 1) {
        berr++;
	return -1;
      }

      // only insert valid strings
      boost::mutex::scoped_lock scoped_lock(mutex);
      prescalers.push_back(s);

      return prescalers.size();
    }

    int PrescaleService::sizePrescale()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);
      return prescalers.size();
    }

    void PrescaleService::getConfig(edm::ParameterSet params)
    {
      std::ostringstream oss;
      unsigned int nss = 0;
      std::string SEPARATOR = " ";
      oss << "0";

      try {

	//        std::cout << "!!! PrescaleService::getConfig list @all_modules" << std::endl;
        std::vector<std::string> pModules = params.getParameter<std::vector<std::string> >("@all_modules");
        for(unsigned int i=0; i<pModules.size(); i++) {
	  //          std::cout << "  index " << i << ", pModules " << pModules[i] << std::endl;
        }

	//        std::cout << "!!! PrescaleService::getConfig list @path" << std::endl;
        std::vector<std::string> pPaths = params.getParameter<std::vector<std::string> >("@paths");
        for(unsigned int i=0; i<pPaths.size(); i++) {
	  //          std::cout << "  index " << i << ", pPaths " << pPaths[i] << std::endl;
        }

	//        std::cout << "!!! PrescaleService::getConfig link modules to paths" << std::endl;
        for(unsigned int i=0; i<pModules.size(); i++) {
	  edm::ParameterSet aa = params.getParameter<edm::ParameterSet>(pModules[i]);
          std::string moduleLabel = aa.getParameter<std::string>("@module_label");
          std::string moduleType = aa.getParameter<std::string>("@module_type");
	  //          std::cout << "!!! label : " << moduleLabel << " type : "  << moduleType << std::endl;
          if(moduleType == "HLTPrescaler") {
	    unsigned int ps = aa.getParameter<unsigned int>("prescaleFactor");
	    //	    std::cout << "!!! label : " << moduleLabel << " type : "  << moduleType << " ps : " << ps << std::endl;
            for(unsigned int j=0; j<pPaths.size(); j++) {
              std::vector<std::string> pPM = params.getParameter<std::vector<std::string> >(pPaths[j]);
              for(unsigned int k=0; k<pPM.size(); k++) {
                if(moduleLabel == pPM[k]) {
		  //                  std::cout << "!!! path " << pPaths[j] << " module " << moduleLabel << " ps " << ps << std::endl;
                  oss << SEPARATOR << pPaths[j] << SEPARATOR << moduleLabel << SEPARATOR << ps;
                  nss++;
                  break;
                }
              }
            }
          }
        }
        if (nss != 0) {
	  //	  std::cout << "!!! PrescaleService::getConfig putPrescale:" << oss.str() << ":" << std::endl;
	  putPrescale(oss.str());
	  //          std::cout << "!!! PrescaleService::getConfig getStatus: " << getStatus() << std::endl;
        }


      }
      catch (edm::Exception &e) {
        bcfg++;
	//        std::cout << "!!! PrescaleService::getConfig caught " << (std::string)e.what() << std::endl;
      }

    }

    void PrescaleService::putHandle(edm::EventProcessor *proc_)
    {
      fu_ = proc_;
    }

    std::string PrescaleService::getStatus()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);

      std::ostringstream oss;
      std::string SEPARATOR = " ";
      oss << prescalers.size();
      oss << SEPARATOR << blsn;
      oss << SEPARATOR << bpath;
      oss << SEPARATOR << bmod;
      oss << SEPARATOR << bfac;
      oss << SEPARATOR << berr;
      oss << SEPARATOR << glow;
      oss << SEPARATOR << pleq;
      oss << SEPARATOR << bang;
      oss << SEPARATOR << triggers.size();
      oss << SEPARATOR << lspmax;
      oss << SEPARATOR << count_;
      oss << SEPARATOR << nops;
      stsstr = oss.str();
      return stsstr;
    }

    std::string PrescaleService::getLs(std::string s)
    {
      int n;
      std::istringstream iss(s);
      iss >> n;
      if (n >= 0) {
	boost::mutex::scoped_lock scoped_lock(mutex);
	std::vector<std::string>::iterator p;
	for(p=triggers.begin(); p != triggers.end(); ) {
	  std::istringstream jss(*p);
	  int n2;
	  jss >> n2;
	  if (n2 < n) {
//	    std::cout << "getLs" << n2 << "<" << n << " erasing " << *p << std::endl;
	    triggers.erase(p);
	    continue;
	  }
	  p++;
	}
      }
      return getLs();
    }
    
    std::string PrescaleService::getLs()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);
      trgstr="";
      for(unsigned int i=0; i<triggers.size(); i++) {
        trgstr += triggers[i];
      //for(unsigned int i=triggers.size(); i>0;) {
      //  trgstr += triggers[--i];
        trgstr += " ";
      }
      return trgstr;
    }

    std::string PrescaleService::getTr()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);

      trstr = " ";
      if (fu_ != 0) {
        fu_->getTriggerReport(tr_);

        // Add an array length indicator so that the resulting string will have a
        // little more readability.
        std::string ARRAY_LEN = "_";
        std::string SEPARATOR = " ";

        std::ostringstream oss;

        //TriggerReport::eventSummary
        oss<<tr_.eventSummary.totalEvents<<SEPARATOR
           <<tr_.eventSummary.totalEventsPassed<<SEPARATOR
           <<tr_.eventSummary.totalEventsFailed<<SEPARATOR;

        //TriggerReport::trigPathSummaries
        oss<<ARRAY_LEN<<tr_.trigPathSummaries.size()<<SEPARATOR;
        for(unsigned int i=0; i<tr_.trigPathSummaries.size(); i++) {
          oss<<tr_.trigPathSummaries[i].bitPosition<<SEPARATOR
             <<tr_.trigPathSummaries[i].timesRun<<SEPARATOR
             <<tr_.trigPathSummaries[i].timesPassed<<SEPARATOR
             <<tr_.trigPathSummaries[i].timesFailed<<SEPARATOR
             <<tr_.trigPathSummaries[i].timesExcept<<SEPARATOR
             <<tr_.trigPathSummaries[i].name<<SEPARATOR;
          //TriggerReport::trigPathSummaries::moduleInPathSummaries
          oss<<ARRAY_LEN<<tr_.trigPathSummaries[i].moduleInPathSummaries.size()<<SEPARATOR;
          for(unsigned int j=0;j<tr_.trigPathSummaries[i].moduleInPathSummaries.size();j++) {
            oss<<tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited<<SEPARATOR
               <<tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed <<SEPARATOR
               <<tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed <<SEPARATOR
               <<tr_.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept <<SEPARATOR
               <<tr_.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<SEPARATOR;
          }
        }

        //TriggerReport::endPathSummaries
        oss<<ARRAY_LEN<<tr_.endPathSummaries.size()<<SEPARATOR;
        for(unsigned int i=0; i<tr_.endPathSummaries.size(); i++) {
          oss<<tr_.endPathSummaries[i].bitPosition<<SEPARATOR
             <<tr_.endPathSummaries[i].timesRun<<SEPARATOR
             <<tr_.endPathSummaries[i].timesPassed<<SEPARATOR
             <<tr_.endPathSummaries[i].timesFailed<<SEPARATOR
             <<tr_.endPathSummaries[i].timesExcept<<SEPARATOR
             <<tr_.endPathSummaries[i].name<<SEPARATOR;
          //TriggerReport::endPathSummaries::moduleInPathSummaries
          oss<<ARRAY_LEN<<tr_.endPathSummaries[i].moduleInPathSummaries.size()<<SEPARATOR;
          for(unsigned int j=0;j<tr_.endPathSummaries[i].moduleInPathSummaries.size();j++) {
            oss<<tr_.endPathSummaries[i].moduleInPathSummaries[j].timesVisited<<SEPARATOR
               <<tr_.endPathSummaries[i].moduleInPathSummaries[j].timesPassed <<SEPARATOR
               <<tr_.endPathSummaries[i].moduleInPathSummaries[j].timesFailed <<SEPARATOR
               <<tr_.endPathSummaries[i].moduleInPathSummaries[j].timesExcept <<SEPARATOR
               <<tr_.endPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<SEPARATOR;
          }
        }

        //TriggerReport::workerSummaries
        oss<<ARRAY_LEN<<tr_.workerSummaries.size()<<SEPARATOR;
        for(unsigned int i=0; i<tr_.workerSummaries.size(); i++) {
          oss<<tr_.workerSummaries[i].timesVisited<<SEPARATOR
             <<tr_.workerSummaries[i].timesRun    <<SEPARATOR
             <<tr_.workerSummaries[i].timesPassed <<SEPARATOR
             <<tr_.workerSummaries[i].timesFailed <<SEPARATOR
             <<tr_.workerSummaries[i].timesExcept <<SEPARATOR
             <<tr_.workerSummaries[i].moduleLabel <<SEPARATOR;
        }
        trstr = oss.str();
      }
      return trstr;
    }



  }
}

