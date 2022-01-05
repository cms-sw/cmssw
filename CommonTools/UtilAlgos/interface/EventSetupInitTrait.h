#ifndef UtilAlgos_EventSetupInitTrait_h
#define UtilAlgos_EventSetupInitTrait_h
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/OrSelector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class EventSetup;
  class Event;
}  // namespace edm

namespace helpers {
  struct NullAndOperand;
}

namespace reco {
  namespace modules {
    /// take no action (default)
    template <typename T>
    struct NoEventSetupInit {
      explicit NoEventSetupInit(edm::ConsumesCollector) {}
      NoEventSetupInit() = delete;
      void init(T&, const edm::Event&, const edm::EventSetup&) {}
    };

    /// implement common interface defined in:
    /// https://twiki.cern.ch/twiki/bin/view/CMS/SelectorInterface
    struct CommonSelectorEventSetupInit {
      explicit CommonSelectorEventSetupInit(edm::ConsumesCollector) {}
      CommonSelectorEventSetupInit() = delete;
      template <typename Selector>
      void init(Selector& selector, const edm::Event& evt, const edm::EventSetup& es) {
        selector.newEvent(evt, es);
      }
    };

    template <typename T>
    struct EventSetupInit {
      typedef NoEventSetupInit<T> type;
    };

    template <typename T1,
              typename T2,
              typename T3 = helpers::NullAndOperand,
              typename T4 = helpers::NullAndOperand,
              typename T5 = helpers::NullAndOperand>
    struct CombinedEventSetupInit {
      explicit CombinedEventSetupInit(edm::ConsumesCollector iC) : t1_(iC), t2_(iC), t3_(iC), t4_(iC), t5_(iC) {}
      template <template <typename, typename, typename, typename, typename> class SelectorT>
      void init(SelectorT<T1, T2, T3, T4, T5>& selector, const edm::Event& evt, const edm::EventSetup& es) {
        t1_.init(selector.s1_, evt, es);
        t2_.init(selector.s2_, evt, es);
        t3_.init(selector.s3_, evt, es);
        t4_.init(selector.s4_, evt, es);
        t5_.init(selector.s5_, evt, es);
      }
      typename EventSetupInit<T1>::type t1_;
      typename EventSetupInit<T2>::type t2_;
      typename EventSetupInit<T3>::type t3_;
      typename EventSetupInit<T4>::type t4_;
      typename EventSetupInit<T5>::type t5_;
    };

    template <typename T1, typename T2, typename T3, typename T4>
    struct CombinedEventSetupInit<T1, T2, T3, T4, helpers::NullAndOperand> {
      explicit CombinedEventSetupInit(edm::ConsumesCollector iC) : t1_(iC), t2_(iC), t3_(iC), t4_(iC) {}
      template <template <typename, typename, typename, typename, typename> class SelectorT>
      void init(SelectorT<T1, T2, T3, T4, helpers::NullAndOperand>& selector,
                const edm::Event& evt,
                const edm::EventSetup& es) {
        t1_.init(selector.s1_, evt, es);
        t2_.init(selector.s2_, evt, es);
        t3_.init(selector.s3_, evt, es);
        t4_.init(selector.s4_, evt, es);
      }
      typename EventSetupInit<T1>::type t1_;
      typename EventSetupInit<T2>::type t2_;
      typename EventSetupInit<T3>::type t3_;
      typename EventSetupInit<T4>::type t4_;
    };

    template <typename T1, typename T2, typename T3>
    struct CombinedEventSetupInit<T1, T2, T3, helpers::NullAndOperand, helpers::NullAndOperand> {
      explicit CombinedEventSetupInit(edm::ConsumesCollector iC) : t1_(iC), t2_(iC), t3_(iC) {}
      template <template <typename, typename, typename, typename, typename> class SelectorT>
      void init(SelectorT<T1, T2, T3, helpers::NullAndOperand, helpers::NullAndOperand>& selector,
                const edm::Event& evt,
                const edm::EventSetup& es) {
        t1_.init(selector.s1_, evt, es);
        t2_.init(selector.s2_, evt, es);
        t3_.init(selector.s3_, evt, es);
      }
      typename EventSetupInit<T1>::type t1_;
      typename EventSetupInit<T2>::type t2_;
      typename EventSetupInit<T3>::type t3_;
    };

    template <typename T1, typename T2>
    struct CombinedEventSetupInit<T1, T2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand> {
      explicit CombinedEventSetupInit(edm::ConsumesCollector iC) : t1_(iC), t2_(iC) {}
      template <template <typename, typename, typename, typename, typename> class SelectorT>
      void init(SelectorT<T1, T2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand>& selector,
                const edm::Event& evt,
                const edm::EventSetup& es) {
        t1_.init(selector.s1_, evt, es);
        t2_.init(selector.s2_, evt, es);
      }
      typename EventSetupInit<T1>::type t1_;
      typename EventSetupInit<T2>::type t2_;
    };

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    struct EventSetupInit<AndSelector<T1, T2, T3, T4, T5> > {
      typedef CombinedEventSetupInit<T1, T2, T3, T4, T5> type;
    };

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    struct EventSetupInit<OrSelector<T1, T2, T3, T4, T5> > {
      typedef CombinedEventSetupInit<T1, T2, T3, T4, T5> type;
    };

  }  // namespace modules
}  // namespace reco

#define EVENTSETUP_STD_INIT(SELECTOR)              \
  namespace reco {                                 \
    namespace modules {                            \
      template <>                                  \
      struct EventSetupInit<SELECTOR> {            \
        typedef CommonSelectorEventSetupInit type; \
      };                                           \
    }                                              \
  }                                                \
  struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T1(SELECTOR)           \
  namespace reco {                                 \
    namespace modules {                            \
      template <typename T1>                       \
      struct EventSetupInit<SELECTOR<T1> > {       \
        typedef CommonSelectorEventSetupInit type; \
      };                                           \
    }                                              \
  }                                                \
  struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T2(SELECTOR)           \
  namespace reco {                                 \
    namespace modules {                            \
      template <typename T1, typename T2>          \
      struct EventSetupInit<SELECTOR<T1, T2> > {   \
        typedef CommonSelectorEventSetupInit type; \
      };                                           \
    }                                              \
  }                                                \
  struct __useless_ignoreme

#define EVENTSETUP_STD_INIT_T3(SELECTOR)               \
  namespace reco {                                     \
    namespace modules {                                \
      template <typename T1, typename T2, typename T3> \
      struct EventSetupInit<SELECTOR<T1, T2, T3> > {   \
        typedef CommonSelectorEventSetupInit type;     \
      };                                               \
    }                                                  \
  }                                                    \
  struct __useless_ignoreme

#endif
