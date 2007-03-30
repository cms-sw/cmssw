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
// $Id: MessageDrop.cc,v 1.4 2006/10/02 20:31:22 marafino Exp $
//

// system include files

#include "boost/thread/tss.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageDrop.h"


using namespace edm;

static boost::thread_specific_ptr<MessageDrop> drops;

edm::Exception * MessageDrop::ex_p = 0;

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

