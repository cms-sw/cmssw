#ifndef FWCore_Framework_ComponentConstructionSentry_h
#define FWCore_Framework_ComponentConstructionSentry_h
//
// Package:     Framework
// Class  :     ComponentConstructionSentry
//
/**\class edm::eventsetup::ComponentConstructionSentry

 Description: Used to send ActivityRegistry signals before and after the construction of an EventSetup component (ESSource or ESProducer)

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 16:56:05 EDT 2005
//

namespace edm {
  namespace eventsetup {
    class EventSetupProvider;
    struct ComponentDescription;

    struct ComponentConstructionSentry {
      ComponentConstructionSentry(EventSetupProvider const& iProvider, ComponentDescription const& iDescription);
      ~ComponentConstructionSentry() noexcept(false);

      //call if the construction call succeeded without throwing an exception.
      void succeeded() { succeeded_ = true; }

    private:
      EventSetupProvider const& provider_;
      ComponentDescription const& description_;
      bool succeeded_ = false;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif