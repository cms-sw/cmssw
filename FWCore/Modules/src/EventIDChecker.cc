// -*- C++ -*-
//
// Package:    Modules
// Class:      EventIDChecker
//
/**\class EventIDChecker EventIDChecker.cc FWCore/Modules/src/EventIDChecker.cc

 Description: Checks that the events passed to it come in the order specified in its configuration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 16 15:42:17 CDT 2009
//
//

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <algorithm>
#include <memory>
#include <vector>

//
// class decleration
//

class EventIDChecker : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit EventIDChecker(edm::ParameterSet const&);
  ~EventIDChecker() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;

  // ----------member data ---------------------------
  std::vector<edm::EventID> ids_;
  unsigned int index_;
  edm::RunNumber_t presentRun_ = 0;
  edm::LuminosityBlockNumber_t presentLumi_ = 0;

  unsigned int multiProcessSequentialEvents_;
  unsigned int numberOfEventsLeftBeforeSearch_;
  bool mustSearch_;
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
EventIDChecker::EventIDChecker(edm::ParameterSet const& iConfig)
    : ids_(iConfig.getUntrackedParameter<std::vector<edm::EventID> >("eventSequence")),
      index_(0),
      multiProcessSequentialEvents_(iConfig.getUntrackedParameter<unsigned int>("multiProcessSequentialEvents")),
      numberOfEventsLeftBeforeSearch_(0),
      mustSearch_(false) {
  //now do what ever initialization is needed
}

EventIDChecker::~EventIDChecker() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

namespace {
  struct CompareWithoutLumi {
    CompareWithoutLumi(edm::EventID const& iThis) : m_this(iThis) {}
    bool operator()(edm::EventID const& iOther) {
      return m_this.run() == iOther.run() && m_this.event() == iOther.event();
    }
    edm::EventID m_this;
  };
}  // namespace

// ------------ method called to for each event  ------------
void EventIDChecker::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  if (mustSearch_) {
    if (0 == numberOfEventsLeftBeforeSearch_) {
      numberOfEventsLeftBeforeSearch_ = multiProcessSequentialEvents_;
      //the event must be after the last event in our list since multicore doesn't go backwards
      std::vector<edm::EventID>::iterator itFind =
          std::find_if(ids_.begin() + index_, ids_.end(), CompareWithoutLumi(iEvent.id()));
      if (itFind == ids_.end()) {
        throw cms::Exception("MissedEvent") << "The event " << iEvent.id() << "is not in the list.\n";
      }
      index_ = itFind - ids_.begin();
    }
    --numberOfEventsLeftBeforeSearch_;
  }

  if (index_ >= ids_.size()) {
    throw cms::Exception("TooManyEvents")
        << "Was passes " << ids_.size() << " EventIDs but have processed more events than that\n";
  }
  if (iEvent.id().run() != ids_[index_].run() || iEvent.id().event() != ids_[index_].event()) {
    throw cms::Exception("WrongEvent") << "Was expecting event " << ids_[index_] << " but was given " << iEvent.id()
                                       << "\n";
  }

  if (presentRun_ != iEvent.run()) {
    throw cms::Exception("MissingRunTransitionForEvent")
        << "at event expected Run " << presentRun_ << " but got " << iEvent.run();
  }
  if (presentLumi_ != iEvent.luminosityBlock()) {
    throw cms::Exception("MissingLuminosityBlockTransitionForEvent")
        << "expected LuminosityBlock " << presentLumi_ << " but got " << iEvent.luminosityBlock();
  }

  ++index_;
}

void EventIDChecker::beginRun(edm::Run const& iRun, edm::EventSetup const&) { presentRun_ = iRun.run(); }
void EventIDChecker::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
  if (presentRun_ != iLumi.run()) {
    throw cms::Exception("WrongRunForLuminosityBlock")
        << "at beginLuminosityBlock expected Run " << presentRun_ << " but got " << iLumi.run();
  }
  presentLumi_ = iLumi.luminosityBlock();
}

void EventIDChecker::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
  if (presentRun_ != iLumi.run()) {
    throw cms::Exception("WrongRunForLuminosityBlock")
        << "at endLuminosityBlock expected Run " << presentRun_ << " but got " << iLumi.run();
  }
  if (presentLumi_ != iLumi.luminosityBlock()) {
    throw cms::Exception("WrongEndLuminosityBlock")
        << "expected LuminosityBlock " << presentLumi_ << " but got " << iLumi.luminosityBlock();
  }
}
void EventIDChecker::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  if (presentRun_ != iRun.run()) {
    throw cms::Exception("WrongEndRun") << "expected Run " << presentRun_ << " but got " << iRun.run();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void EventIDChecker::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EventIDChecker::endJob() {}

// ------------ method called once each job for validation
void EventIDChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::EventID> >("eventSequence");
  desc.addUntracked<unsigned int>("multiProcessSequentialEvents", 0U);
  descriptions.add("eventIDChecker", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventIDChecker);
