// -*- C++ -*-
//
// Package:     Services
// Class  :     log4cplus
// 
//
// Original Author:  Jim Kowalkowski
// $Id: MLlog4cplus.cc,v 1.5 2006/02/20 01:58:20 wmtan Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageService/interface/NamedDestination.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Message2log4cplus/src/ELlog4cplus.h"

#include <iostream>

using namespace edm;

namespace ML {
  
  // class ELlog4cplus exists

  class MLlog4cplus
  {
  public:
    MLlog4cplus(const ParameterSet&,ActivityRegistry&);
    ~MLlog4cplus();
      
    void postBeginJob();
    void postEndJob();
      
    void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
    void postEventProcessing(const Event&, const EventSetup&);
      
    void preModule(const ModuleDescription&);
    void postModule(const ModuleDescription&);
  private:
    edm::EventID curr_event_;
  };
}

using namespace ML;

  MLlog4cplus::MLlog4cplus(const ParameterSet& iPS, ActivityRegistry&iRegistry)
  {
    // we may want these in the future, but probably not, since the
    // MessageLogger service is supposed to deal with that stuff anyway (JBK)

    // iRegistry.watchPostBeginJob(this,&MLlog4cplus::postBeginJob);
    // iRegistry.watchPostEndJob(this,&MLlog4cplus::postEndJob);
    
    // iRegistry.watchPreProcessEvent(this,&MLlog4cplus::preEventProcessing);
    // iRegistry.watchPostProcessEvent(this,&MLlog4cplus::postEventProcessing);
    
    // iRegistry.watchPreModule(this,&MLlog4cplus::preModule);
    // iRegistry.watchPostModule(this,&MLlog4cplus::postModule);


    // pseudo-code:
    // get message logger message queue (singleton)
    // make new ELlog4cplus object using parameterset set information
    // make a message with opcode NEWDEST
    // send message (NEWDEST,ELdest*)


    // we should first get a handle to the MessageLogger service to
    // ensure that it is initialized before we are (JBK)
    // edm::Service<edm::MessageLogger> handle;

    ELlog4cplus * dest_p = new ELlog4cplus;
    edm::service::NamedDestination * ndest = new edm::service::NamedDestination ( "log4cplus", dest_p ); 
    edm::MessageLoggerQ::EXT(ndest);
  }


  MLlog4cplus::~MLlog4cplus()
  {
  }

  void MLlog4cplus::postBeginJob()
  {
  }

  void MLlog4cplus::postEndJob()
  {
  }

  void MLlog4cplus::preEventProcessing(const edm::EventID& iID,
				       const edm::Timestamp& iTime)
  {
  }

  void MLlog4cplus::postEventProcessing(const Event& e, const EventSetup&)
  {
  }

  void MLlog4cplus::preModule(const ModuleDescription&)
  {
  }

  void MLlog4cplus::postModule(const ModuleDescription& desc)
  {
  }



using ML::MLlog4cplus;
DEFINE_FWK_SERVICE(MLlog4cplus)

