// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     MessageDrop
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  M. Fischler and Jim Kowalkowsi
//         Created:  Tues Feb 14 16:38:19 CST 2006
// $Id: MessageDrop.cc,v 1.2 2006/02/16 17:52:06 fischler Exp $
//

// system include files

#include "boost/thread/tss.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageDrop.h"


using namespace edm;

static boost::thread_specific_ptr<MessageDrop> drops;

MessageDrop *
MessageDrop::instance()
{
  MessageDrop* drop = drops.get();
  if(drop==0) { 
    drops.reset(new MessageDrop);
    drop=drops.get(); 
  }
  return drop;
}

  bool edm::isDebugEnabled() {
    return ( edm::MessageDrop::instance()->debugEnabled );
  }

  bool edm::isInfoEnabled() {
    return( edm::MessageDrop::instance()->infoEnabled );
  }

  bool edm::isWarningEnabled() {
    return( edm::MessageDrop::instance()->warningEnabled );
  }

