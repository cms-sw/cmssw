// -*- C++ -*-
//
// Package:     FWCoreIntegration
// Class  :     DoodadESSource
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 14:39:39 EDT 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/CoreFramework/interface/ESProducer.h"
#include "FWCore/CoreFramework/interface/SourceFactory.h"


#include "FWCore/FWCoreIntegration/src/GadgetRcd.h"
#include "FWCore/FWCoreIntegration/src/Doodad.h"

namespace edmreftest {
class DoodadESSource :
   public edm::eventsetup::EventSetupRecordIntervalFinder, 
   public edm::eventsetup::ESProducer
{
   
public:
   DoodadESSource(const edm::ParameterSet& );
   
   std::auto_ptr<Doodad> produce( const GadgetRcd& ) ;
   
protected:
   
   virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                const edm::Timestamp& iTime, 
                                edm::ValidityInterval& iInterval);
   
private:
   DoodadESSource( const DoodadESSource& ); // stop default
   
   const DoodadESSource& operator=( const DoodadESSource& ); // stop default
   
   // ---------- member data --------------------------------
   unsigned int nCalls_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DoodadESSource::DoodadESSource(const edm::ParameterSet& )
: nCalls_(0) {
   this->findingRecord<GadgetRcd>();
   setWhatProduced(this);
}

//DoodadESSource::~DoodadESSource()
//{
//}

//
// member functions
//

std::auto_ptr<Doodad> 
DoodadESSource::produce( const GadgetRcd& ) {
   std::auto_ptr<Doodad> data( new Doodad() );
   data->a = nCalls_;
   ++nCalls_;
   return data;
}


void 
DoodadESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                const edm::Timestamp& iTime, 
                                edm::ValidityInterval& iInterval) {
   //Be valid for 3 time steps
   unsigned long newTime = (iTime.value() - 1 ) - ((iTime.value() - 1 ) %3) +1;
   unsigned long endTime = newTime + 2;
   iInterval = edm::ValidityInterval( edm::Timestamp( newTime),
                                      edm::Timestamp(endTime) );
}

//
// const member functions
//

//
// static member functions
//
}
using namespace edmreftest;

DEFINE_FWK_EVENTSETUP_SOURCE(DoodadESSource)

