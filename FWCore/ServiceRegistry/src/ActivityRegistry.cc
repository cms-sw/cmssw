// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ActivityRegistry
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep  6 10:26:49 EDT 2005
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//ActivityRegistry::ActivityRegistry() {
//}

// ActivityRegistry::ActivityRegistry(ActivityRegistry const& rhs) {
//    // do actual copying here;
// }

//ActivityRegistry::~ActivityRegistry() {
//}

//
// assignment operators
//
// ActivityRegistry const& ActivityRegistry::operator=(ActivityRegistry const& rhs) {
//   //An exception safe implementation is
//   ActivityRegistry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
namespace edm {
  namespace {
    template <typename T>
    inline void copySlotsToFrom(T& iTo, T& iFrom) {
      for (auto& slot : iFrom.slots()) {
        iTo.connect(slot);
      }
    }

    template <typename T>
    inline void copySlotsToFromReverse(T& iTo, T& iFrom) {
      // This handles service slots that are supposed to be in reverse
      // order of construction. Copying new ones in is a little
      // tricky.  Here is an example of what follows
      // slots in iTo before  4 3 2 1  and copy in slots in iFrom 8 7 6 5.
      // reverse iFrom  5 6 7 8
      // then do the copy to front 8 7 6 5 4 3 2 1

      auto slotsFrom = iFrom.slots();

      std::reverse(slotsFrom.begin(), slotsFrom.end());

      for (auto& slotFrom : slotsFrom) {
        iTo.connect_front(slotFrom);
      }
    }
  }  // namespace

  namespace signalslot {
    void throwObsoleteSignalException() {
      throw cms::Exception("ConnectedToObsoleteServiceSignal")
          << "A Service has connected to an obsolete ActivityRegistry signal.";
    }
  }  // namespace signalslot

  void ActivityRegistry::connectGlobals(ActivityRegistry& iOther) {
    preallocateSignal_.connect(std::cref(iOther.preallocateSignal_));
    postBeginJobSignal_.connect(std::cref(iOther.postBeginJobSignal_));
    preEndJobSignal_.connect(std::cref(iOther.preEndJobSignal_));
    postEndJobSignal_.connect(std::cref(iOther.postEndJobSignal_));

    jobFailureSignal_.connect(std::cref(iOther.jobFailureSignal_));

    preSourceSignal_.connect(std::cref(iOther.preSourceSignal_));
    postSourceSignal_.connect(std::cref(iOther.postSourceSignal_));

    preSourceLumiSignal_.connect(std::cref(iOther.preSourceLumiSignal_));
    postSourceLumiSignal_.connect(std::cref(iOther.postSourceLumiSignal_));

    preSourceRunSignal_.connect(std::cref(iOther.preSourceRunSignal_));
    postSourceRunSignal_.connect(std::cref(iOther.postSourceRunSignal_));

    preOpenFileSignal_.connect(std::cref(iOther.preOpenFileSignal_));
    postOpenFileSignal_.connect(std::cref(iOther.postOpenFileSignal_));

    preCloseFileSignal_.connect(std::cref(iOther.preCloseFileSignal_));
    postCloseFileSignal_.connect(std::cref(iOther.postCloseFileSignal_));

    preSourceConstructionSignal_.connect(std::cref(iOther.preSourceConstructionSignal_));
    postSourceConstructionSignal_.connect(std::cref(iOther.postSourceConstructionSignal_));

    preStreamEarlyTerminationSignal_.connect(std::cref(iOther.preStreamEarlyTerminationSignal_));
    preGlobalEarlyTerminationSignal_.connect(std::cref(iOther.preGlobalEarlyTerminationSignal_));
    preSourceEarlyTerminationSignal_.connect(std::cref(iOther.preSourceEarlyTerminationSignal_));
  }

  void ActivityRegistry::connectLocals(ActivityRegistry& iOther) {
    preBeginJobSignal_.connect(std::cref(iOther.preBeginJobSignal_));

    preModuleBeginStreamSignal_.connect(std::cref(iOther.preModuleBeginStreamSignal_));
    postModuleBeginStreamSignal_.connect(std::cref(iOther.postModuleBeginStreamSignal_));

    preModuleEndStreamSignal_.connect(std::cref(iOther.preModuleEndStreamSignal_));
    postModuleEndStreamSignal_.connect(std::cref(iOther.postModuleEndStreamSignal_));

    preGlobalBeginRunSignal_.connect(std::cref(iOther.preGlobalBeginRunSignal_));
    postGlobalBeginRunSignal_.connect(std::cref(iOther.postGlobalBeginRunSignal_));

    preGlobalEndRunSignal_.connect(std::cref(iOther.preGlobalEndRunSignal_));
    postGlobalEndRunSignal_.connect(std::cref(iOther.postGlobalEndRunSignal_));

    preGlobalWriteRunSignal_.connect(std::cref(iOther.preGlobalWriteRunSignal_));
    postGlobalWriteRunSignal_.connect(std::cref(iOther.postGlobalWriteRunSignal_));

    preStreamBeginRunSignal_.connect(std::cref(iOther.preStreamBeginRunSignal_));
    postStreamBeginRunSignal_.connect(std::cref(iOther.postStreamBeginRunSignal_));

    preStreamEndRunSignal_.connect(std::cref(iOther.preStreamEndRunSignal_));
    postStreamEndRunSignal_.connect(std::cref(iOther.postStreamEndRunSignal_));

    preGlobalBeginLumiSignal_.connect(std::cref(iOther.preGlobalBeginLumiSignal_));
    postGlobalBeginLumiSignal_.connect(std::cref(iOther.postGlobalBeginLumiSignal_));

    preGlobalEndLumiSignal_.connect(std::cref(iOther.preGlobalEndLumiSignal_));
    postGlobalEndLumiSignal_.connect(std::cref(iOther.postGlobalEndLumiSignal_));

    preGlobalWriteLumiSignal_.connect(std::cref(iOther.preGlobalWriteLumiSignal_));
    postGlobalWriteLumiSignal_.connect(std::cref(iOther.postGlobalWriteLumiSignal_));

    preStreamBeginLumiSignal_.connect(std::cref(iOther.preStreamBeginLumiSignal_));
    postStreamBeginLumiSignal_.connect(std::cref(iOther.postStreamBeginLumiSignal_));

    preStreamEndLumiSignal_.connect(std::cref(iOther.preStreamEndLumiSignal_));
    postStreamEndLumiSignal_.connect(std::cref(iOther.postStreamEndLumiSignal_));

    preEventSignal_.connect(std::cref(iOther.preEventSignal_));
    postEventSignal_.connect(std::cref(iOther.postEventSignal_));

    prePathEventSignal_.connect(std::cref(iOther.prePathEventSignal_));
    postPathEventSignal_.connect(std::cref(iOther.postPathEventSignal_));

    //preProcessEventSignal_.connect(std::cref(iOther.preProcessEventSignal_));
    //postProcessEventSignal_.connect(std::cref(iOther.postProcessEventSignal_));

    //preBeginRunSignal_.connect(std::cref(iOther.preBeginRunSignal_));
    //postBeginRunSignal_.connect(std::cref(iOther.postBeginRunSignal_));

    //preEndRunSignal_.connect(std::cref(iOther.preEndRunSignal_));
    //postEndRunSignal_.connect(std::cref(iOther.postEndRunSignal_));

    //preBeginLumiSignal_.connect(std::cref(iOther.preBeginLumiSignal_));
    //postBeginLumiSignal_.connect(std::cref(iOther.postBeginLumiSignal_));

    //preEndLumiSignal_.connect(std::cref(iOther.preEndLumiSignal_));
    //postEndLumiSignal_.connect(std::cref(iOther.postEndLumiSignal_));

    //preProcessPathSignal_.connect(std::cref(iOther.preProcessPathSignal_));
    //postProcessPathSignal_.connect(std::cref(iOther.postProcessPathSignal_));

    //prePathBeginRunSignal_.connect(std::cref(iOther.prePathBeginRunSignal_));
    //postPathBeginRunSignal_.connect(std::cref(iOther.postPathBeginRunSignal_));

    //prePathEndRunSignal_.connect(std::cref(iOther.prePathEndRunSignal_));
    //postPathEndRunSignal_.connect(std::cref(iOther.postPathEndRunSignal_));

    //prePathBeginLumiSignal_.connect(std::cref(iOther.prePathBeginLumiSignal_));
    //postPathBeginLumiSignal_.connect(std::cref(iOther.postPathBeginLumiSignal_));

    //prePathEndLumiSignal_.connect(std::cref(iOther.prePathEndLumiSignal_));
    //postPathEndLumiSignal_.connect(std::cref(iOther.postPathEndLumiSignal_));

    preModuleConstructionSignal_.connect(std::cref(iOther.preModuleConstructionSignal_));
    postModuleConstructionSignal_.connect(std::cref(iOther.postModuleConstructionSignal_));

    preModuleBeginJobSignal_.connect(std::cref(iOther.preModuleBeginJobSignal_));
    postModuleBeginJobSignal_.connect(std::cref(iOther.postModuleBeginJobSignal_));

    preModuleEndJobSignal_.connect(std::cref(iOther.preModuleEndJobSignal_));
    postModuleEndJobSignal_.connect(std::cref(iOther.postModuleEndJobSignal_));

    preModuleEventPrefetchingSignal_.connect(std::cref(iOther.preModuleEventPrefetchingSignal_));
    postModuleEventPrefetchingSignal_.connect(std::cref(iOther.postModuleEventPrefetchingSignal_));

    preModuleEventSignal_.connect(std::cref(iOther.preModuleEventSignal_));
    postModuleEventSignal_.connect(std::cref(iOther.postModuleEventSignal_));

    preModuleEventAcquireSignal_.connect(std::cref(iOther.preModuleEventAcquireSignal_));
    postModuleEventAcquireSignal_.connect(std::cref(iOther.postModuleEventAcquireSignal_));

    preModuleEventDelayedGetSignal_.connect(std::cref(iOther.preModuleEventDelayedGetSignal_));
    postModuleEventDelayedGetSignal_.connect(std::cref(iOther.postModuleEventDelayedGetSignal_));

    preEventReadFromSourceSignal_.connect(std::cref(iOther.preEventReadFromSourceSignal_));
    postEventReadFromSourceSignal_.connect(std::cref(iOther.postEventReadFromSourceSignal_));

    preModuleStreamBeginRunSignal_.connect(std::cref(iOther.preModuleStreamBeginRunSignal_));
    postModuleStreamBeginRunSignal_.connect(std::cref(iOther.postModuleStreamBeginRunSignal_));

    preModuleStreamEndRunSignal_.connect(std::cref(iOther.preModuleStreamEndRunSignal_));
    postModuleStreamEndRunSignal_.connect(std::cref(iOther.postModuleStreamEndRunSignal_));

    preModuleStreamBeginLumiSignal_.connect(std::cref(iOther.preModuleStreamBeginLumiSignal_));
    postModuleStreamBeginLumiSignal_.connect(std::cref(iOther.postModuleStreamBeginLumiSignal_));

    preModuleStreamEndLumiSignal_.connect(std::cref(iOther.preModuleStreamEndLumiSignal_));
    postModuleStreamEndLumiSignal_.connect(std::cref(iOther.postModuleStreamEndLumiSignal_));

    preModuleGlobalBeginRunSignal_.connect(std::cref(iOther.preModuleGlobalBeginRunSignal_));
    postModuleGlobalBeginRunSignal_.connect(std::cref(iOther.postModuleGlobalBeginRunSignal_));

    preModuleGlobalEndRunSignal_.connect(std::cref(iOther.preModuleGlobalEndRunSignal_));
    postModuleGlobalEndRunSignal_.connect(std::cref(iOther.postModuleGlobalEndRunSignal_));

    preModuleGlobalBeginLumiSignal_.connect(std::cref(iOther.preModuleGlobalBeginLumiSignal_));
    postModuleGlobalBeginLumiSignal_.connect(std::cref(iOther.postModuleGlobalBeginLumiSignal_));

    preModuleGlobalEndLumiSignal_.connect(std::cref(iOther.preModuleGlobalEndLumiSignal_));
    postModuleGlobalEndLumiSignal_.connect(std::cref(iOther.postModuleGlobalEndLumiSignal_));

    preModuleWriteRunSignal_.connect(std::cref(iOther.preModuleWriteRunSignal_));
    postModuleWriteRunSignal_.connect(std::cref(iOther.postModuleWriteRunSignal_));

    preModuleWriteLumiSignal_.connect(std::cref(iOther.preModuleWriteLumiSignal_));
    postModuleWriteLumiSignal_.connect(std::cref(iOther.postModuleWriteLumiSignal_));

    //preModuleSignal_.connect(std::cref(iOther.preModuleSignal_));
    //postModuleSignal_.connect(std::cref(iOther.postModuleSignal_));

    //preModuleBeginRunSignal_.connect(std::cref(iOther.preModuleBeginRunSignal_));
    //postModuleBeginRunSignal_.connect(std::cref(iOther.postModuleBeginRunSignal_));

    //preModuleEndRunSignal_.connect(std::cref(iOther.preModuleEndRunSignal_));
    //postModuleEndRunSignal_.connect(std::cref(iOther.postModuleEndRunSignal_));

    //preModuleBeginLumiSignal_.connect(std::cref(iOther.preModuleBeginLumiSignal_));
    //postModuleBeginLumiSignal_.connect(std::cref(iOther.postModuleBeginLumiSignal_));

    //preModuleEndLumiSignal_.connect(std::cref(iOther.preModuleEndLumiSignal_));
    //postModuleEndLumiSignal_.connect(std::cref(iOther.postModuleEndLumiSignal_));

    preLockEventSetupGetSignal_.connect(std::cref(iOther.preLockEventSetupGetSignal_));
    postLockEventSetupGetSignal_.connect(std::cref(iOther.postLockEventSetupGetSignal_));
    postEventSetupGetSignal_.connect(std::cref(iOther.postEventSetupGetSignal_));
  }

  void ActivityRegistry::connect(ActivityRegistry& iOther) {
    connectGlobals(iOther);
    connectLocals(iOther);
  }

  void ActivityRegistry::connectToSubProcess(ActivityRegistry& iOther) {
    connectGlobals(iOther);       // child sees parents global signals
    iOther.connectLocals(*this);  // parent see childs global signals
  }

  void ActivityRegistry::copySlotsFrom(ActivityRegistry& iOther) {
    copySlotsToFrom(preallocateSignal_, iOther.preallocateSignal_);
    copySlotsToFrom(preBeginJobSignal_, iOther.preBeginJobSignal_);
    copySlotsToFrom(postBeginJobSignal_, iOther.postBeginJobSignal_);
    copySlotsToFromReverse(preEndJobSignal_, iOther.preEndJobSignal_);
    copySlotsToFromReverse(postEndJobSignal_, iOther.postEndJobSignal_);

    copySlotsToFromReverse(jobFailureSignal_, iOther.jobFailureSignal_);

    copySlotsToFrom(preSourceSignal_, iOther.preSourceSignal_);
    copySlotsToFromReverse(postSourceSignal_, iOther.postSourceSignal_);

    copySlotsToFrom(preSourceLumiSignal_, iOther.preSourceLumiSignal_);
    copySlotsToFromReverse(postSourceLumiSignal_, iOther.postSourceLumiSignal_);

    copySlotsToFrom(preSourceRunSignal_, iOther.preSourceRunSignal_);
    copySlotsToFromReverse(postSourceRunSignal_, iOther.postSourceRunSignal_);

    copySlotsToFrom(preOpenFileSignal_, iOther.preOpenFileSignal_);
    copySlotsToFromReverse(postOpenFileSignal_, iOther.postOpenFileSignal_);

    copySlotsToFrom(preCloseFileSignal_, iOther.preCloseFileSignal_);
    copySlotsToFromReverse(postCloseFileSignal_, iOther.postCloseFileSignal_);

    copySlotsToFrom(preModuleBeginStreamSignal_, iOther.preModuleBeginStreamSignal_);
    copySlotsToFromReverse(postModuleBeginStreamSignal_, iOther.postModuleBeginStreamSignal_);

    copySlotsToFrom(preModuleEndStreamSignal_, iOther.preModuleEndStreamSignal_);
    copySlotsToFromReverse(postModuleEndStreamSignal_, iOther.postModuleEndStreamSignal_);

    copySlotsToFrom(preGlobalBeginRunSignal_, iOther.preGlobalBeginRunSignal_);
    copySlotsToFromReverse(postGlobalBeginRunSignal_, iOther.postGlobalBeginRunSignal_);

    copySlotsToFrom(preGlobalEndRunSignal_, iOther.preGlobalEndRunSignal_);
    copySlotsToFromReverse(postGlobalEndRunSignal_, iOther.postGlobalEndRunSignal_);

    copySlotsToFrom(preGlobalWriteRunSignal_, iOther.preGlobalWriteRunSignal_);
    copySlotsToFromReverse(postGlobalWriteRunSignal_, iOther.postGlobalWriteRunSignal_);

    copySlotsToFrom(preStreamBeginRunSignal_, iOther.preStreamBeginRunSignal_);
    copySlotsToFromReverse(postStreamBeginRunSignal_, iOther.postStreamBeginRunSignal_);

    copySlotsToFrom(preStreamEndRunSignal_, iOther.preStreamEndRunSignal_);
    copySlotsToFromReverse(postStreamEndRunSignal_, iOther.postStreamEndRunSignal_);

    copySlotsToFrom(preGlobalBeginLumiSignal_, iOther.preGlobalBeginLumiSignal_);
    copySlotsToFromReverse(postGlobalBeginLumiSignal_, iOther.postGlobalBeginLumiSignal_);

    copySlotsToFrom(preGlobalEndLumiSignal_, iOther.preGlobalEndLumiSignal_);
    copySlotsToFromReverse(postGlobalEndLumiSignal_, iOther.postGlobalEndLumiSignal_);

    copySlotsToFrom(preGlobalWriteLumiSignal_, iOther.preGlobalWriteLumiSignal_);
    copySlotsToFromReverse(postGlobalWriteLumiSignal_, iOther.postGlobalWriteLumiSignal_);

    copySlotsToFrom(preStreamBeginLumiSignal_, iOther.preStreamBeginLumiSignal_);
    copySlotsToFromReverse(postStreamBeginLumiSignal_, iOther.postStreamBeginLumiSignal_);

    copySlotsToFrom(preStreamEndLumiSignal_, iOther.preStreamEndLumiSignal_);
    copySlotsToFromReverse(postStreamEndLumiSignal_, iOther.postStreamEndLumiSignal_);

    copySlotsToFrom(preEventSignal_, iOther.preEventSignal_);
    copySlotsToFromReverse(postEventSignal_, iOther.postEventSignal_);

    copySlotsToFrom(prePathEventSignal_, iOther.prePathEventSignal_);
    copySlotsToFromReverse(postPathEventSignal_, iOther.postPathEventSignal_);

    copySlotsToFrom(preStreamEarlyTerminationSignal_, iOther.preStreamEarlyTerminationSignal_);
    copySlotsToFrom(preGlobalEarlyTerminationSignal_, iOther.preGlobalEarlyTerminationSignal_);
    copySlotsToFrom(preSourceEarlyTerminationSignal_, iOther.preSourceEarlyTerminationSignal_);

    /*
    copySlotsToFrom(preProcessEventSignal_, iOther.preProcessEventSignal_);
    copySlotsToFromReverse(postProcessEventSignal_, iOther.postProcessEventSignal_);

    copySlotsToFrom(preBeginRunSignal_, iOther.preBeginRunSignal_);
    copySlotsToFromReverse(postBeginRunSignal_, iOther.postBeginRunSignal_);

    copySlotsToFrom(preEndRunSignal_, iOther.preEndRunSignal_);
    copySlotsToFromReverse(postEndRunSignal_, iOther.postEndRunSignal_);

    copySlotsToFrom(preBeginLumiSignal_, iOther.preBeginLumiSignal_);
    copySlotsToFromReverse(postBeginLumiSignal_, iOther.postBeginLumiSignal_);

    copySlotsToFrom(preEndLumiSignal_, iOther.preEndLumiSignal_);
    copySlotsToFromReverse(postEndLumiSignal_, iOther.postEndLumiSignal_);

    copySlotsToFrom(preProcessPathSignal_, iOther.preProcessPathSignal_);
    copySlotsToFromReverse(postProcessPathSignal_, iOther.postProcessPathSignal_);

    copySlotsToFrom(prePathBeginRunSignal_, iOther.prePathBeginRunSignal_);
    copySlotsToFromReverse(postPathBeginRunSignal_, iOther.postPathBeginRunSignal_);

    copySlotsToFrom(prePathEndRunSignal_, iOther.prePathEndRunSignal_);
    copySlotsToFromReverse(postPathEndRunSignal_, iOther.postPathEndRunSignal_);

    copySlotsToFrom(prePathBeginLumiSignal_, iOther.prePathBeginLumiSignal_);
    copySlotsToFromReverse(postPathBeginLumiSignal_, iOther.postPathBeginLumiSignal_);

    copySlotsToFrom(prePathEndLumiSignal_, iOther.prePathEndLumiSignal_);
    copySlotsToFromReverse(postPathEndLumiSignal_, iOther.postPathEndLumiSignal_);
*/
    copySlotsToFrom(preModuleConstructionSignal_, iOther.preModuleConstructionSignal_);
    copySlotsToFromReverse(postModuleConstructionSignal_, iOther.postModuleConstructionSignal_);

    copySlotsToFrom(preModuleBeginJobSignal_, iOther.preModuleBeginJobSignal_);
    copySlotsToFromReverse(postModuleBeginJobSignal_, iOther.postModuleBeginJobSignal_);

    copySlotsToFrom(preModuleEndJobSignal_, iOther.preModuleEndJobSignal_);
    copySlotsToFromReverse(postModuleEndJobSignal_, iOther.postModuleEndJobSignal_);

    copySlotsToFrom(preModuleEventPrefetchingSignal_, iOther.preModuleEventPrefetchingSignal_);
    copySlotsToFromReverse(postModuleEventPrefetchingSignal_, iOther.postModuleEventPrefetchingSignal_);

    copySlotsToFrom(preModuleEventSignal_, iOther.preModuleEventSignal_);
    copySlotsToFromReverse(postModuleEventSignal_, iOther.postModuleEventSignal_);

    copySlotsToFrom(preModuleEventAcquireSignal_, iOther.preModuleEventAcquireSignal_);
    copySlotsToFromReverse(postModuleEventAcquireSignal_, iOther.postModuleEventAcquireSignal_);

    copySlotsToFrom(preModuleEventDelayedGetSignal_, iOther.preModuleEventDelayedGetSignal_);
    copySlotsToFromReverse(postModuleEventDelayedGetSignal_, iOther.postModuleEventDelayedGetSignal_);

    copySlotsToFrom(preEventReadFromSourceSignal_, iOther.preEventReadFromSourceSignal_);
    copySlotsToFromReverse(postEventReadFromSourceSignal_, iOther.postEventReadFromSourceSignal_);

    copySlotsToFrom(preModuleStreamBeginRunSignal_, iOther.preModuleStreamBeginRunSignal_);
    copySlotsToFromReverse(postModuleStreamBeginRunSignal_, iOther.postModuleStreamBeginRunSignal_);

    copySlotsToFrom(preModuleStreamEndRunSignal_, iOther.preModuleStreamEndRunSignal_);
    copySlotsToFromReverse(postModuleStreamEndRunSignal_, iOther.postModuleStreamEndRunSignal_);

    copySlotsToFrom(preModuleStreamBeginLumiSignal_, iOther.preModuleStreamBeginLumiSignal_);
    copySlotsToFromReverse(postModuleStreamBeginLumiSignal_, iOther.postModuleStreamBeginLumiSignal_);

    copySlotsToFrom(preModuleStreamEndLumiSignal_, iOther.preModuleStreamEndLumiSignal_);
    copySlotsToFromReverse(postModuleStreamEndLumiSignal_, iOther.postModuleStreamEndLumiSignal_);

    copySlotsToFrom(preModuleGlobalBeginRunSignal_, iOther.preModuleGlobalBeginRunSignal_);
    copySlotsToFromReverse(postModuleGlobalBeginRunSignal_, iOther.postModuleGlobalBeginRunSignal_);

    copySlotsToFrom(preModuleGlobalEndRunSignal_, iOther.preModuleGlobalEndRunSignal_);
    copySlotsToFromReverse(postModuleGlobalEndRunSignal_, iOther.postModuleGlobalEndRunSignal_);

    copySlotsToFrom(preModuleGlobalBeginLumiSignal_, iOther.preModuleGlobalBeginLumiSignal_);
    copySlotsToFromReverse(postModuleGlobalBeginLumiSignal_, iOther.postModuleGlobalBeginLumiSignal_);

    copySlotsToFrom(preModuleGlobalEndLumiSignal_, iOther.preModuleGlobalEndLumiSignal_);
    copySlotsToFromReverse(postModuleGlobalEndLumiSignal_, iOther.postModuleGlobalEndLumiSignal_);

    copySlotsToFrom(preModuleWriteRunSignal_, iOther.preModuleWriteRunSignal_);
    copySlotsToFromReverse(postModuleWriteRunSignal_, iOther.postModuleWriteRunSignal_);

    copySlotsToFrom(preModuleWriteLumiSignal_, iOther.preModuleWriteLumiSignal_);
    copySlotsToFromReverse(postModuleWriteLumiSignal_, iOther.postModuleWriteLumiSignal_);

    /*
    copySlotsToFrom(preModuleSignal_, iOther.preModuleSignal_);
    copySlotsToFromReverse(postModuleSignal_, iOther.postModuleSignal_);

    copySlotsToFrom(preModuleBeginRunSignal_, iOther.preModuleBeginRunSignal_);
    copySlotsToFromReverse(postModuleBeginRunSignal_, iOther.postModuleBeginRunSignal_);

    copySlotsToFrom(preModuleEndRunSignal_, iOther.preModuleEndRunSignal_);
    copySlotsToFromReverse(postModuleEndRunSignal_, iOther.postModuleEndRunSignal_);

    copySlotsToFrom(preModuleBeginLumiSignal_, iOther.preModuleBeginLumiSignal_);
    copySlotsToFromReverse(postModuleBeginLumiSignal_, iOther.postModuleBeginLumiSignal_);

    copySlotsToFrom(preModuleEndLumiSignal_, iOther.preModuleEndLumiSignal_);
    copySlotsToFromReverse(postModuleEndLumiSignal_, iOther.postModuleEndLumiSignal_);
     */
    copySlotsToFrom(preSourceConstructionSignal_, iOther.preSourceConstructionSignal_);
    copySlotsToFromReverse(postSourceConstructionSignal_, iOther.postSourceConstructionSignal_);

    copySlotsToFrom(preLockEventSetupGetSignal_, iOther.preLockEventSetupGetSignal_);
    copySlotsToFromReverse(postLockEventSetupGetSignal_, iOther.postLockEventSetupGetSignal_);
    copySlotsToFromReverse(postEventSetupGetSignal_, iOther.postEventSetupGetSignal_);
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}  // namespace edm
