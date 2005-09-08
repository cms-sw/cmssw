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
// $Id$
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
Tracer::Tracer(const ParameterSet&, ActivityRegistry&iRegistry)
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
   std::cout <<"++ Job started"<<std::endl;
}
void 
Tracer::postEndJob()
{
   std::cout <<"++ Job ended"<<std::endl;
}

void 
Tracer::preEventProcessing(const edm::EventID& iID, const edm::Timestamp& iTime)
{
   std::cout <<"++++ processing event:"<< iID<<" time:"<<iTime.value()<< std::endl;
}
void 
Tracer::postEventProcessing(const Event&, const EventSetup&)
{
   std::cout <<"++++ finished event:"<<std::endl;
}

void 
Tracer::preModule(const ModuleDescription& iDescription)
{
   std::cout <<"++++++ module:" <<iDescription.moduleLabel_<<std::endl;
}

void 
Tracer::postModule(const ModuleDescription&)
{
   std::cout <<"++++++ finished"<<std::endl;
}

//
// const member functions
//

//
// static member functions
//
