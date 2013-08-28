// -*- C++ -*-
//
// Package:     Modules
// Class  :     IterateNTimesLooper
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 11 11:16:14 EDT 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  class IterateNTimesLooper : public EDLooper {

    public:
      IterateNTimesLooper(ParameterSet const&);
      virtual ~IterateNTimesLooper();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void startingNewLoop(unsigned int) override;
      virtual Status duringLoop(Event const&, EventSetup const&) override;
      virtual Status endOfLoop(EventSetup const&, unsigned int) override;

    private:
      IterateNTimesLooper(IterateNTimesLooper const&); // stop default

      IterateNTimesLooper const& operator=(IterateNTimesLooper const&); // stop default

      // ---------- member data --------------------------------
      unsigned int max_;
      unsigned int times_;
      bool shouldStop_;
  };

  //
  //
  // constructors and destructor
  //
  IterateNTimesLooper::IterateNTimesLooper(ParameterSet const& iConfig) :
    max_(iConfig.getParameter<unsigned int>("nTimes")),
    times_(0),
    shouldStop_(false) {
  }

  // IterateNTimesLooper::IterateNTimesLooper(IterateNTimesLooper const& rhs) {
  //    // do actual copying here;
  // }

  IterateNTimesLooper::~IterateNTimesLooper() {
  }

  //
  // assignment operators
  //
  // IterateNTimesLooper const& IterateNTimesLooper::operator=(IterateNTimesLooper const& rhs) {
  //   //An exception safe implementation is
  //   IterateNTimesLooper temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  void
  IterateNTimesLooper::startingNewLoop(unsigned int iIteration) {
    times_ = iIteration;
    if(iIteration >= max_) {
      shouldStop_ = true;
    }
  }

  EDLooper::Status
  IterateNTimesLooper::duringLoop(Event const&, EventSetup const&) {
    return shouldStop_ ? kStop : kContinue;
  }

  EDLooper::Status
  IterateNTimesLooper::endOfLoop(EventSetup const&, unsigned int /*iCounter*/) {
    ++times_;
    return (times_ < max_) ? kContinue : kStop;
  }
}

using edm::IterateNTimesLooper;
DEFINE_FWK_LOOPER(IterateNTimesLooper);
