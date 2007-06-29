// -*- C++ -*-
//
// Package:     Services
// Class  :     PrescaleServices
// 
// Implementation:
//     Cache and make prescale factors available online.
//
// Current revision: $Revision: 1.2 $
// On branch: $Name: V00-00-02 $
// Latest change by $Author: wmtan $ on $Date: 2007/04/23 23:54:11 $ 
//

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

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
 
      LogDebug("PrescaleService") << "PrescaleService::PrescaleService";

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
	  ostringstream oss;
	  string ARRAY_LEN = "_";
	  string SEPARATOR = " ";
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

    int PrescaleService::getPrescale(unsigned int ls, string module)
    {
      boost::mutex::scoped_lock scoped_lock(mutex);

      if (ls < lsgmax) {
	glow++;
      } else {
	lsgmax = ls;
      }

      int j = prescalers.size()-1;
      for( ; j>=0; j--) {
//	cout << "getPrescale j " << j << endl;
	unsigned int n; 
	istringstream iss(prescalers[j]);
	iss >> n;
//	cout << "getPrescale n " << n << " ls " << ls << endl;
	if (ls >= n) {
//	    cout << "getPrescale n  " << n << " <= " << " ls " << ls << endl;
	    break;
	}
      }
      if (j < 0) j = 0;

      unsigned int i = j;
      unsigned int n, m;
      string a, b;
      istringstream iss(prescalers[i]);
      iss >> n;
      while (!(iss.rdstate() & istringstream::goodbit)) {
	iss >> a >> b >> m; 
//	cout << "getPrescale " << n << "==" << ls << " a " << a << " b " << b << " m " << m << endl;
	if (module == b) {
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

    int PrescaleService::putPrescale(string s)
    {

      istringstream iss(s);
      unsigned int i, n, m;
      string a, b;
      
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



    void PrescaleService::putHandle(edm::EventProcessor *proc_)
    {
      fu_ = proc_;
    }

    string PrescaleService::getStatus()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);

      ostringstream oss;
      string SEPARATOR = " ";
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
      stsstr = oss.str();
      return stsstr;
    }

    string PrescaleService::getLs(string s)
    {
      int n;
      istringstream iss(s);
      iss >> n;
      if (n >= 0) {
	boost::mutex::scoped_lock scoped_lock(mutex);
	vector<string>::iterator p;
	for(p=triggers.begin(); p != triggers.end(); ) {
	  istringstream jss(*p);
	  int n2;
	  jss >> n2;
	  if (n2 < n) {
//	    cout << "getLs" << n2 << "<" << n << " erasing " << *p << endl;
	    triggers.erase(p);
	    continue;
	  }
	  p++;
	}
      }
      return getLs();
    }
    
    string PrescaleService::getLs()
    {
      boost::mutex::scoped_lock scoped_lock(mutex);
      trgstr="";
      for(unsigned int i=0; i<triggers.size(); i++) {
        trgstr += triggers[i];
        trgstr += " ";
      }
      return trgstr;
    }

  }
}

