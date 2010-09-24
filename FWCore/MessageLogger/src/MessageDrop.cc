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
// $Id: MessageDrop.cc,v 1.8 2010/02/08 23:55:16 chrjones Exp $
//

// system include files

#include "boost/thread/tss.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// Change Log
//
// 1 12/13/07 mf     	the static drops had been file-global level; moved it
//		     	into the instance() method to cure a 24-byte memory
//			leak reported by valgrind. Suggested by MP.
//
// 2 9/23/10 mf		Variables supporting situations where no thresholds are
//                      low enough to react to LogDebug (or info, or warning)

using namespace edm;


edm::Exception * MessageDrop::ex_p = 0;
bool MessageDrop::debugEnabled=true;
bool MessageDrop::infoEnabled=true;
bool MessageDrop::warningEnabled=true;
// The following are false at initialization (in case configure is not done)
// and are set true at the start of configure_ordinary_destinations, 
// but are set false once a destination is thresholded to react to the 
// corresponding severity: 
bool MessageDrop::debugAlwaysSuppressed=false;		// change log 2
bool MessageDrop::infoAlwaysSuppressed=false;	 	// change log 2
bool MessageDrop::warningAlwaysSuppressed=false; 	// change log 2

MessageDrop *
MessageDrop::instance()
{
  static boost::thread_specific_ptr<MessageDrop> drops;
  MessageDrop* drop = drops.get();
  if(drop==0) { 
    drops.reset(new MessageDrop);
    drop=drops.get(); 
  }
  return drop;
}

unsigned char MessageDrop::messageLoggerScribeIsRunning = 0;
