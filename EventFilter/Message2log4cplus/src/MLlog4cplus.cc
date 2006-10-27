// -*- C++ -*-
//
// Package:     Services
// Class  :     log4cplus
// 
//
// Original Author:  Jim Kowalkowski
// $Id: MLlog4cplus.cc,v 1.2 2006/06/13 14:36:22 meschi Exp $
//

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageService/interface/NamedDestination.h"
#include "DataFormats/Common/interface/EventID.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Message2log4cplus/src/ELlog4cplus.h"
#include "EventFilter/Message2log4cplus/interface/MLlog4cplus.h"

#include <iostream>


using namespace edm;


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

    dest_p = new ELlog4cplus;
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
  void MLlog4cplus::setAppl(xdaq::Application *app)
  {
    if(dest_p!=0)dest_p->setAppl(app);
  }

using ML::MLlog4cplus;
DEFINE_FWK_SERVICE(MLlog4cplus);

