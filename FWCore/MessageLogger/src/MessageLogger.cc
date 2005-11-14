// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. Brown, M. Fischler
//         Created:  Fri Nov 11 16:42:39 CST 2005
// $Id: MessageService.cc,v 1.1 2005/11/11 23:04:37 fischler Exp $
//

// system include files

// user include files
#include "FWCore/MessageLogger/src/MessageService.h"

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
MessageService::MessageService(const ParameterSet& iPS, ActivityRegistry&iRegistry)
{
   iRegistry.watchPostBeginJob(this,&MessageService::postBeginJob);
   iRegistry.watchPostEndJob(this,&MessageService::postEndJob);

   iRegistry.watchPreProcessEvent(this,&MessageService::preEventProcessing);
   iRegistry.watchPostProcessEvent(this,&MessageService::postEventProcessing);

   iRegistry.watchPreModule(this,&MessageService::preModule);
   iRegistry.watchPostModule(this,&MessageService::postModule);
}

// MessageService::MessageService(const MessageService& rhs)
// {
//    // do actual copying here;
// }

//MessageService::~MessageService()
//{
//}

//
// assignment operators
//
// const MessageService& MessageService::operator=(const MessageService& rhs)
// {
//   //An exception safe implementation is
//   MessageService temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
MessageService::postBeginJob()
{
   std::cout << " Job started"<<std::endl;
}
void 
MessageService::postEndJob()
{
   std::cout << " Job ended"<<std::endl;
}

void 
MessageService::preEventProcessing(const edm::EventID& iID, const edm::Timestamp& iTime)
{
   std::cout <<" processing event:"<< iID<<" time:"<<iTime.value()<< std::endl;
}
void 
MessageService::postEventProcessing(const Event&, const EventSetup&)
{
   std::cout <<" finished event:"<<std::endl;
}

void 
MessageService::preModule(const ModuleDescription& iDescription)
{
   std::cout <<" module:" <<iDescription.moduleLabel_<<std::endl;
}

void 
MessageService::postModule(const ModuleDescription&)
{
   std::cout <<" finished"<<std::endl;
}

//
// const member functions
//

//
// static member functions
//
