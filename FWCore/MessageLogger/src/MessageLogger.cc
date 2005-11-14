// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. Brown, M. Fischler
//         Created:  Fri Nov 11 16:42:39 CST 2005
// $Id: MessageLogger.cc,v 1.1 2005/11/14 16:36:49 fischler Exp $
//

// system include files

// user include files
#include "FWCore/MessageLogger/src/MessageLogger.h"

#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MessageLogger::MessageLogger(const ParameterSet& iPS, ActivityRegistry&iRegistry)
{
   iRegistry.watchPostBeginJob(this,&MessageLogger::postBeginJob);
   iRegistry.watchPostEndJob(this,&MessageLogger::postEndJob);

   iRegistry.watchPreProcessEvent(this,&MessageLogger::preEventProcessing);
   iRegistry.watchPostProcessEvent(this,&MessageLogger::postEventProcessing);

   iRegistry.watchPreModule(this,&MessageLogger::preModule);
   iRegistry.watchPostModule(this,&MessageLogger::postModule);
}

// MessageLogger::MessageLogger(const MessageLogger& rhs)
// {
//    // do actual copying here;
// }

//MessageLogger::~MessageLogger()
//{
//}

//
// assignment operators
//
// const MessageLogger& MessageLogger::operator=(const MessageLogger& rhs)
// {
//   //An exception safe implementation is
//   MessageLogger temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
MessageLogger::postBeginJob()
{
   std::cout << " MessageLogger Job started"<<std::endl;
}
void 
MessageLogger::postEndJob()
{
   std::cout << " Job ended"<<std::endl;
}

void 
MessageLogger::preEventProcessing(const edm::EventID& iID, const edm::Timestamp& iTime)
{
   std::cout <<" processing event:"<< iID<<" time:"<<iTime.value()<< std::endl;
}
void 
MessageLogger::postEventProcessing(const Event&, const EventSetup&)
{
   std::cout <<" finished event:"<<std::endl;
}

void 
MessageLogger::preModule(const ModuleDescription& iDescription)
{
   std::cout <<" module:" <<iDescription.moduleLabel_<<std::endl;
}

void 
MessageLogger::postModule(const ModuleDescription&)
{
   std::cout <<" finished"<<std::endl;
}

//
// const member functions
//

//
// static member functions
//
