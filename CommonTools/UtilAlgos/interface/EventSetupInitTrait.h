#ifndef UtilAlgos_EventSetupInitTrait_h
#define UtilAlgos_EventSetupInitTrait_h
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/OrSelector.h"

namespace edm {
  class EventSetup;
  class Event;
}

namespace helpers {
  struct NullAndOperand;
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

    template<typename T1, typename T2, typename T3 = helpers::NullAndOperand, 
      typename T4 = helpers::NullAndOperand, typename T5 = helpers::NullAndOperand>
    struct CombinedEventSetupInit {
      template<template<typename, typename, typename, typename, typename> class SelectorT>
      static void init(SelectorT<T1, T2, T3, T4, T5>& selector,
		       const edm::Event & evt,
		       const edm::EventSetup& es) {
        EventSetupInit<T1>::type::init(selector.s1_, evt, es);
	EventSetupInit<T2>::type::init(selector.s2_, evt, es);
	EventSetupInit<T3>::type::init(selector.s3_, evt, es);
	EventSetupInit<T4>::type::init(selector.s4_, evt, es);
	EventSetupInit<T5>::type::init(selector.s5_, evt, es);
      }
    };
    
    template<typename T1, typename T2, typename T3, typename T4>
    struct CombinedEventSetupInit<T1, T2, T3, T4, helpers::NullAndOperand> {
      template<template<typename, typename, typename, typename, typename> class SelectorT>
      static void init(SelectorT<T1, T2, T3, T4, helpers::NullAndOperand>& selector,
		       const edm::Event & evt,
		       const edm::EventSetup& es) {
        EventSetupInit<T1>::type::init(selector.s1_, evt, es);
	EventSetupInit<T2>::type::init(selector.s2_, evt, es);
	EventSetupInit<T3>::type::init(selector.s3_, evt, es);
	EventSetupInit<T4>::type::init(selector.s4_, evt, es);
      }
    };
    
    template<typename T1, typename T2, typename T3>
    struct CombinedEventSetupInit<T1, T2, T3, helpers::NullAndOperand, helpers::NullAndOperand> {
      template<template<typename, typename, typename, typename, typename> class SelectorT>
      static void init(SelectorT<T1, T2, T3, helpers::NullAndOperand, helpers::NullAndOperand>& selector,
		       const edm::Event & evt,
		       const edm::EventSetup& es) {
        EventSetupInit<T1>::type::init(selector.s1_, evt, es);
	EventSetupInit<T2>::type::init(selector.s2_, evt, es);
	EventSetupInit<T3>::type::init(selector.s3_, evt, es);
      }
    };
    
    template<typename T1, typename T2>
    struct CombinedEventSetupInit<T1, T2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand> {
      template<template<typename, typename, typename, typename, typename> class SelectorT>
      static void init(SelectorT<T1, T2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand>& selector,
		       const edm::Event & evt,
		       const edm::EventSetup& es) {
        EventSetupInit<T1>::type::init(selector.s1_, evt, es);
	EventSetupInit<T2>::type::init(selector.s2_, evt, es);
      }
    };
    
    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    struct EventSetupInit<AndSelector<T1, T2, T3, T4, T5> > {
      typedef CombinedEventSetupInit<T1, T2, T3, T4, T5> type;
    };

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    struct EventSetupInit<OrSelector<T1, T2, T3, T4, T5> > {
      typedef CombinedEventSetupInit<T1, T2, T3, T4, T5> type;
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

