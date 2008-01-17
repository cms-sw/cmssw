#ifndef UtilAlgos_EventSetupInitTrait_h
#define UtilAlgos_EventSetupInitTrait_h

namespace edm {
  class EventSetup;
  class Event;
}

namespace reco {
  namespace modules {
    /// take no action (default)
    template<typename T>
    struct NoEventSetupInit {
      static void init(T &, const edm::Event&, const edm::EventSetup&) { }
    };

    /// implement common interface defined in:
    /// https://twiki.cern.ch/twiki/bin/view/CMS/SelectorInterface
    struct CommonSelectorEventSetupInit {
      template<typename Selector>
      static void init( Selector & selector,
			const edm::Event & evt,
			const edm::EventSetup& es ) { 
	selector.newEvent(evt, es);
      }
    };

    template<typename T>
    struct EventSetupInit {
      typedef NoEventSetupInit<T> type;
    };
  }
}

#define EVENTSETUP_STD_INIT(SELECTOR) \
namespace reco { \
  namespace modules { \
    template<> \
    struct EventSetupInit<SELECTOR> { \
      typedef CommonSelectorEventSetupInit type; \
    }; \
  } \
} \

#endif
