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
struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T1(SELECTOR) \
namespace reco { \
  namespace modules { \
    template<typename T1> \
    struct EventSetupInit<SELECTOR<T1> > {	 \
      typedef CommonSelectorEventSetupInit type; \
    }; \
  } \
} \
struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T2(SELECTOR) \
namespace reco { \
  namespace modules { \
    template<typename T1, typename T2>			 \
    struct EventSetupInit<SELECTOR<T1, T2> > {		 \
      typedef CommonSelectorEventSetupInit type; \
    }; \
  } \
} \
struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T3(SELECTOR) \
namespace reco { \
  namespace modules { \
    template<typename T1, typename T2, typename T3>		 \
    struct EventSetupInit<SELECTOR<T1, T2, T3> > {		 \
      typedef CommonSelectorEventSetupInit type; \
    }; \
  } \
} \
struct __useless_ignoreme


#endif
