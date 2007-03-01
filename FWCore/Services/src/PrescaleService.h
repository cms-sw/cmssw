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
// Latest change by $Author$ at $Date$
//


#ifndef Services_PrescaleService_h
#define Services_PrescaleService_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/TriggerReport.h"

#include "boost/thread/mutex.hpp"

using namespace std;

#include <vector>
#include <iostream>


namespace edm {
  namespace service {

    // cache structure used to store prescalers
    struct element {
      string path;
      string module;
      unsigned int value;
    };
    typedef element element;

    struct cache {
      unsigned int ls;
      unsigned int touch;
      vector<element> modules;
    };
    typedef cache cache;

    // cache class
    class Cache {
      boost::mutex mutex;
      vector<cache> members;
      unsigned int updcnt;
      unsigned int badcnt;
      unsigned int baddec;
      unsigned int getcnt;
      unsigned int getbad;
      int lastls;

    public:
      Cache() { 
	updcnt = 0;
	badcnt = 0;
	baddec = 0;
	getcnt = 0;
	getbad = 0;
        lastls = 0;
      }

      // add complete member to cache
      // input string format = "LS# path module prescaler ... path module prescaler" 
      void add(const string s) {
        vector<string> tokens;
        string delims = " ";

        // tokenize everything
	string::size_type last = s.find_first_not_of(delims, 0);
	string::size_type posn = s.find_first_of(delims, last);
        while (string::npos != posn || string::npos != last) {
	  tokens.push_back(s.substr(last, posn - last));
	  last = s.find_first_not_of(delims, posn);
	  posn = s.find_first_of(delims, last);
        }

	// lock to update counters and cache
	boost::mutex::scoped_lock scoped_lock(mutex);

        // validate tokens
        int value;
        if (tokens.size() < 4 || (tokens.size()%3) != 1) badcnt++;
        for (unsigned int i=0; i<tokens.size(); i += 3) {
	  if(sscanf(tokens[i].c_str(), "%d", &value) != 1) baddec++;
        }

        // cache tokens
        int ls;
	updcnt++;
        sscanf(tokens[0].c_str(), "%d", &ls);
        for (unsigned int i=1; i<tokens.size(); i += 3) {
	  sscanf(tokens[i+2].c_str(), "%d", &value);
	  add(ls, tokens[i], tokens[i+1], value);
        }
	lastls = ls;

      }

      // add single element to cache member - should not lock the mutex
      void add(unsigned int ls, string path, string module, unsigned int value) {
	vector<cache>::iterator p;
	for(p=members.begin(); p != members.end(); p++) {
	  if (p->ls == ls) {
	    element e;
	    e.path = path;
	    e.module = module;
	    e.value = value;
	    p->modules.push_back(e);
	    return;
	  }
	}

	cache c;
	c.ls = ls;
	c.touch = 0;
	element e;
	e.path = path;
        e.module = module;
        e.value = value;
	c.modules.push_back(e);
	members.push_back(c);
	for(p=members.begin(); p != members.end(); p++) p->touch += 1;
	if (members.size() > 5) {
	  for(p=members.begin(); p != members.end(); p++) {
	    if (p->touch > 5) {
	      members.erase(p);
	    }
	  }
	}
      }

      // get the prescale value associated with LS# and module name
      int get(unsigned int ls, string module) {
	boost::mutex::scoped_lock scoped_lock(mutex);

	vector<cache>::iterator p;
	for(p=members.begin(); p != members.end(); p++) {
	  if (ls == p->ls) {
	    vector<element>::iterator q;
	    for(q=p->modules.begin(); q != p->modules.end(); q++) {
	      if (module == q->module) return q->value;
	    }
	    break;
	  }
	}
	return lastls;
      }

      // prints cache status to stdout
      void show() {
	boost::mutex::scoped_lock scoped_lock(mutex);

	vector<cache>::iterator p;
	for(p=members.begin(); p != members.end(); p++) {
	  cout << " member " << p->ls;
	  vector<element>::iterator q;
	  for(q=p->modules.begin(); q != p->modules.end(); q++) {
	    cout << " p " << q->path << " m " << q->module << " v " << q->value;
	  }
	  cout << endl;
	}
      }

      // return number of members in the cache
      int size() {
        return members.size();
      }

    };

    class PrescaleService
    {
      edm::EventID curr_event_;
      int count_;
      Cache store;
      edm::EventProcessor *fu_;
      edm::TriggerReport tr_;
      string trs_;
      int lsnr_;

    public:
      PrescaleService(const ParameterSet&,ActivityRegistry&);
      ~PrescaleService();
      
      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      unsigned int getPrescale(unsigned int ls, string module);
      int putPrescale(string s);
      int sizePrescale();
      void putHandle(edm::EventProcessor *proc_);
      string getTriggerCounters();
	
    };
  }
}


#endif
