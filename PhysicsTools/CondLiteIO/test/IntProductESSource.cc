// -*- C++ -*-
//
// Package:     Integration
// Class  :     IntProductESSource
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

#include "PhysicsTools/CondLiteIO/test/IntProductRecord.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace edmtest {
  class IntProductESSource : public edm::EventSetupRecordIntervalFinder, public edm::ESProducer {
  public:
    IntProductESSource(const edm::ParameterSet&);

    std::unique_ptr<edmtest::IntProduct> produce(const IntProductRecord&);

  protected:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                        const edm::IOVSyncValue& iTime,
                        edm::ValidityInterval& iInterval) override;

  private:
    IntProductESSource(const IntProductESSource&) = delete;  // stop default

    const IntProductESSource& operator=(const IntProductESSource&) = delete;  // stop default

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
  IntProductESSource::IntProductESSource(const edm::ParameterSet&) : nCalls_(0) {
    this->findingRecord<IntProductRecord>();
    setWhatProduced(this);
  }

  //IntProductESSource::~IntProductESSource()
  //{
  //}

  //
  // member functions
  //

  std::unique_ptr<edmtest::IntProduct> IntProductESSource::produce(const IntProductRecord&) {
    auto data = std::make_unique<edmtest::IntProduct>();
    data->value = nCalls_;
    ++nCalls_;
    return data;
  }

  void IntProductESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                          const edm::IOVSyncValue& iTime,
                                          edm::ValidityInterval& iInterval) {
    //Be valid for 3 runs
    edm::EventID newTime = edm::EventID(1, 0, 0);
    edm::EventID endTime = edm::EventID(4, 0, 0);
    iInterval = edm::ValidityInterval(edm::IOVSyncValue(newTime), edm::IOVSyncValue(endTime));
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}  // namespace edmtest
using namespace edmtest;

DEFINE_FWK_EVENTSETUP_SOURCE(IntProductESSource);
