// -*- C++ -*-
//
// Package:    DummyLooper
// Class:      DummyLooper
//
/**\class DummyLooper DummyLooper.h FWCore/DummyLooper/interface/DummyLooper.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Valentin Kuznetsov
//         Created:  Tue Jul 18 10:17:05 EDT 2006
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

//
// class decleration
//
using namespace edm::eventsetup::test;

class DummyLooper : public edm::ESProducerLooper {
public:
  DummyLooper(const edm::ParameterSet&);
  ~DummyLooper() override;

  typedef std::shared_ptr<DummyData> ReturnType;
  typedef std::shared_ptr<DummyData const> ConstReturnType;

  ReturnType produce(const DummyRecord&);

  void startingNewLoop(unsigned int) override {}
  Status duringLoop(const edm::Event&, const edm::EventSetup&) override { return issueStop_ ? kStop : kContinue; }
  Status endOfLoop(const edm::EventSetup&, unsigned int) override {
    (data_->value_)++;
    ++counter_;
    return counter_ == 2 ? kStop : kContinue;
  }

private:
  // ----------member data ---------------------------
  ConstReturnType data() const { return get_underlying_safe(data_); }
  ReturnType& data() { return get_underlying_safe(data_); }

  edm::propagate_const<ReturnType> data_;
  int counter_;
  bool issueStop_;
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
DummyLooper::DummyLooper(const edm::ParameterSet& iConfig)
    : data_(new DummyData(iConfig.getUntrackedParameter<int>("value"))),
      counter_(0),
      issueStop_(iConfig.getUntrackedParameter<bool>("issueStop", false)) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  //now do what ever other initialization is needed
}

DummyLooper::~DummyLooper() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
DummyLooper::ReturnType DummyLooper::produce(const DummyRecord&) { return data(); }

//define this as a plug-in
DEFINE_FWK_LOOPER(DummyLooper);
