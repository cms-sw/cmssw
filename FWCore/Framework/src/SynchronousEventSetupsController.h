#ifndef FWCore_Framework_SynchronousEventSetupsController_h
#define FWCore_Framework_SynchronousEventSetupsController_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     SynchronousEventSetupsController
//
/** \class edm::eventsetup::SynchronousEventSetupsController

 Description: Manages a group of EventSetups which can share components.

 Usage:
    Useful for unit testing parts of the EventSetup system

*/
//
// Original Authors:  Chris Jones, David Dagenhart
//          Created:  Wed Jan 12 14:30:42 CST 2011
//

#include "FWCore/Framework/interface/EventSetupsController.h"

#include "tbb/task_group.h"
#include "tbb/global_control.h"

namespace edm {

  namespace eventsetup {

    class SynchronousEventSetupsController {
    public:
      SynchronousEventSetupsController();
      ~SynchronousEventSetupsController();

      SynchronousEventSetupsController(SynchronousEventSetupsController const&) = delete;
      SynchronousEventSetupsController const& operator=(SynchronousEventSetupsController const&) = delete;
      SynchronousEventSetupsController(SynchronousEventSetupsController&&) = delete;
      SynchronousEventSetupsController const& operator=(SynchronousEventSetupsController&&) = delete;

      std::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&,
                                                       ActivityRegistry*,
                                                       ParameterSet const* eventSetupPset = nullptr);

      // Version to use when IOVs are not allowed to run concurrently
      void eventSetupForInstance(IOVSyncValue const&);

    private:
      tbb::global_control globalControl_;
      tbb::task_group taskGroup_;
      EventSetupsController controller_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
