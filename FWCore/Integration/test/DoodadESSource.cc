// -*- C++ -*-
//
// Package:     Integration
// Class  :     DoodadESSource
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 14:39:39 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Integration/test/GadgetRcd.h"
#include "FWCore/Integration/test/Doodad.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {
class DoodadESSource :
   public edm::EventSetupRecordIntervalFinder, 
   public edm::ESProducer
{
   
public:
   DoodadESSource(edm::ParameterSet const& pset);
   
   std::auto_ptr<Doodad> produce(const GadgetRcd&) ;

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
   
   virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval);
   
private:
   DoodadESSource(const DoodadESSource&); // stop default
   
   const DoodadESSource& operator=(const DoodadESSource&); // stop default
   
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
DoodadESSource::DoodadESSource(edm::ParameterSet const& pset)
: nCalls_(0) {

  if (pset.getUntrackedParameter<bool>("test", true)) {
     throw edm::Exception(edm::errors::Configuration, "Something is wrong with ESSource validation\n")
       << "Or the test configuration parameter was set true (it should never be true unless you want this exception)\n";
   }

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
DoodadESSource::produce(const GadgetRcd&) {
   std::auto_ptr<Doodad> data(new Doodad());
   data->a = nCalls_;
   ++nCalls_;
   return data;
}

void
DoodadESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addOptional<std::string>("appendToDataLabel");
  desc.addOptionalUntracked<std::string>("test2");
  desc.addUntracked<bool>("test", false)->
    setComment("This parameter exists only to test the parameter set validation for ESSources"); 
  descriptions.add("DoodadESSource", desc);
}

void 
DoodadESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval) {
   //Be valid for 3 runs 
   edm::EventID newTime = edm::EventID((iTime.eventID().run() - 1) - ((iTime.eventID().run() - 1) %3) +1, 1, 1);
   edm::EventID endTime = newTime.nextRun(1).nextRun(1).nextRun(1).previousRunLastEvent(1);
   iInterval = edm::ValidityInterval(edm::IOVSyncValue(newTime),
                                      edm::IOVSyncValue(endTime));
}

//
// const member functions
//

//
// static member functions
//
}
using namespace edmtest;

DEFINE_FWK_EVENTSETUP_SOURCE(DoodadESSource);

