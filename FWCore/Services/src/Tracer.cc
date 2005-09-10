// -*- C++ -*-
//
// Package:     Services
// Class  :     Tracer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 14:17:58 EDT 2005
// $Id: Tracer.cc,v 1.2 2005/09/08 20:37:18 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Services/src/Tracer.h"

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
Tracer::Tracer(const ParameterSet& iPS, ActivityRegistry&iRegistry):
indention_(iPS.getUntrackedParameter<std::string>("indention","++"))
{
   iRegistry.watchPostBeginJob(this,&Tracer::postBeginJob);
   iRegistry.watchPostEndJob(this,&Tracer::postEndJob);

   iRegistry.watchPreProcessEvent(this,&Tracer::preEventProcessing);
   iRegistry.watchPostProcessEvent(this,&Tracer::postEventProcessing);

   iRegistry.watchPreModule(this,&Tracer::preModule);
   iRegistry.watchPostModule(this,&Tracer::postModule);
}

// Tracer::Tracer(const Tracer& rhs)
// {
//    // do actual copying here;
// }

//Tracer::~Tracer()
//{
//}

//
// assignment operators
//
// const Tracer& Tracer::operator=(const Tracer& rhs)
// {
//   //An exception safe implementation is
//   Tracer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
Tracer::postBeginJob()
{
   std::cout <<indention_<<" Job started"<<std::endl;
}
void 
Tracer::postEndJob()
{
   std::cout <<indention_<<" Job ended"<<std::endl;
}

void 
Tracer::preEventProcessing(const edm::EventID& iID, const edm::Timestamp& iTime)
{
   std::cout <<indention_<<indention_<<" processing event:"<< iID<<" time:"<<iTime.value()<< std::endl;
}
void 
Tracer::postEventProcessing(const Event&, const EventSetup&)
{
   std::cout <<indention_<<indention_<<" finished event:"<<std::endl;
}

void 
Tracer::preModule(const ModuleDescription& iDescription)
{
   std::cout <<indention_<<indention_<<indention_<<" module:" <<iDescription.moduleLabel_<<std::endl;
}

void 
Tracer::postModule(const ModuleDescription&)
{
   std::cout <<indention_<<indention_<<indention_<<" finished"<<std::endl;
}

//
// const member functions
//

//
// static member functions
//
