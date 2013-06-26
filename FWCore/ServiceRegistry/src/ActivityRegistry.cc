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
    template<typename T>
    inline
    void
    copySlotsToFrom(T& iTo, T& iFrom) {
      for(auto& slot: iFrom.slots()) {
        iTo.connect(slot);
      }
    }

    template<typename T>
    inline
    void
    copySlotsToFromReverse(T& iTo, T& iFrom) {
      // This handles service slots that are supposed to be in reverse
      // order of construction. Copying new ones in is a little
      // tricky.  Here is an example of what follows
      // slots in iTo before  4 3 2 1  and copy in slots in iFrom 8 7 6 5.
      // reverse iFrom  5 6 7 8
      // then do the copy to front 8 7 6 5 4 3 2 1

      auto slotsFrom = iFrom.slots();

      std::reverse(slotsFrom.begin(), slotsFrom.end());

      for(auto& slotFrom: slotsFrom) {
        iTo.connect_front(slotFrom);
      }
    }
  }

  void
  ActivityRegistry::connectGlobals(ActivityRegistry& iOther) {

     postBeginJobSignal_.connect(std::cref(iOther.postBeginJobSignal_));
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

     preProcessEventSignal_.connect(std::cref(iOther.preProcessEventSignal_));
     postProcessEventSignal_.connect(std::cref(iOther.postProcessEventSignal_));

     preBeginRunSignal_.connect(std::cref(iOther.preBeginRunSignal_));
     postBeginRunSignal_.connect(std::cref(iOther.postBeginRunSignal_));

     preEndRunSignal_.connect(std::cref(iOther.preEndRunSignal_));
     postEndRunSignal_.connect(std::cref(iOther.postEndRunSignal_));

     preBeginLumiSignal_.connect(std::cref(iOther.preBeginLumiSignal_));
     postBeginLumiSignal_.connect(std::cref(iOther.postBeginLumiSignal_));

     preEndLumiSignal_.connect(std::cref(iOther.preEndLumiSignal_));
     postEndLumiSignal_.connect(std::cref(iOther.postEndLumiSignal_));

     preForkReleaseResourcesSignal_.connect(std::cref(iOther.preForkReleaseResourcesSignal_));
     postForkReacquireResourcesSignal_.connect(std::cref(iOther.postForkReacquireResourcesSignal_));
  }

  void
  ActivityRegistry::connectLocals(ActivityRegistry& iOther) {

     preProcessPathSignal_.connect(std::cref(iOther.preProcessPathSignal_));
     postProcessPathSignal_.connect(std::cref(iOther.postProcessPathSignal_));

     prePathBeginRunSignal_.connect(std::cref(iOther.prePathBeginRunSignal_));
     postPathBeginRunSignal_.connect(std::cref(iOther.postPathBeginRunSignal_));

     prePathEndRunSignal_.connect(std::cref(iOther.prePathEndRunSignal_));
     postPathEndRunSignal_.connect(std::cref(iOther.postPathEndRunSignal_));

     prePathBeginLumiSignal_.connect(std::cref(iOther.prePathBeginLumiSignal_));
     postPathBeginLumiSignal_.connect(std::cref(iOther.postPathBeginLumiSignal_));

     prePathEndLumiSignal_.connect(std::cref(iOther.prePathEndLumiSignal_));
     postPathEndLumiSignal_.connect(std::cref(iOther.postPathEndLumiSignal_));

     preModuleSignal_.connect(std::cref(iOther.preModuleSignal_));
     postModuleSignal_.connect(std::cref(iOther.postModuleSignal_));

     preModuleBeginRunSignal_.connect(std::cref(iOther.preModuleBeginRunSignal_));
     postModuleBeginRunSignal_.connect(std::cref(iOther.postModuleBeginRunSignal_));

     preModuleEndRunSignal_.connect(std::cref(iOther.preModuleEndRunSignal_));
     postModuleEndRunSignal_.connect(std::cref(iOther.postModuleEndRunSignal_));

     preModuleBeginLumiSignal_.connect(std::cref(iOther.preModuleBeginLumiSignal_));
     postModuleBeginLumiSignal_.connect(std::cref(iOther.postModuleBeginLumiSignal_));

     preModuleEndLumiSignal_.connect(std::cref(iOther.preModuleEndLumiSignal_));
     postModuleEndLumiSignal_.connect(std::cref(iOther.postModuleEndLumiSignal_));

     preModuleConstructionSignal_.connect(std::cref(iOther.preModuleConstructionSignal_));
     postModuleConstructionSignal_.connect(std::cref(iOther.postModuleConstructionSignal_));

     preModuleBeginJobSignal_.connect(std::cref(iOther.preModuleBeginJobSignal_));
     postModuleBeginJobSignal_.connect(std::cref(iOther.postModuleBeginJobSignal_));

     preModuleEndJobSignal_.connect(std::cref(iOther.preModuleEndJobSignal_));
     postModuleEndJobSignal_.connect(std::cref(iOther.postModuleEndJobSignal_));
  }

  void
  ActivityRegistry::connect(ActivityRegistry& iOther) {
    connectGlobals(iOther);
    connectLocals(iOther);
  }

  void
  ActivityRegistry::connectToSubProcess(ActivityRegistry& iOther) {
    connectGlobals(iOther);
    iOther.connectLocals(*this);
  }

  void
  ActivityRegistry::copySlotsFrom(ActivityRegistry& iOther) {
    copySlotsToFrom(postBeginJobSignal_, iOther.postBeginJobSignal_);
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

    copySlotsToFrom(preModuleConstructionSignal_, iOther.preModuleConstructionSignal_);
    copySlotsToFromReverse(postModuleConstructionSignal_, iOther.postModuleConstructionSignal_);

    copySlotsToFrom(preModuleBeginJobSignal_, iOther.preModuleBeginJobSignal_);
    copySlotsToFromReverse(postModuleBeginJobSignal_, iOther.postModuleBeginJobSignal_);

    copySlotsToFrom(preModuleEndJobSignal_, iOther.preModuleEndJobSignal_);
    copySlotsToFromReverse(postModuleEndJobSignal_, iOther.postModuleEndJobSignal_);

    copySlotsToFrom(preSourceConstructionSignal_, iOther.preSourceConstructionSignal_);
    copySlotsToFromReverse(postSourceConstructionSignal_, iOther.postSourceConstructionSignal_);

    copySlotsToFrom(preForkReleaseResourcesSignal_, iOther.preForkReleaseResourcesSignal_);
    copySlotsToFromReverse(postForkReacquireResourcesSignal_, iOther.postForkReacquireResourcesSignal_);
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}
