#ifndef UtilAlgos_EventSetupInitTrait_h
#define UtilAlgos_EventSetupInitTrait_h

namespace edm {
  class EventSetup;
}

namespace reco {
  namespace modules {
    template<typename T>
    struct NoEventSetupInit {
      static void init( T &, const edm::EventSetup& es ) { }
    };

    template<typename T>
    struct EventSetupInit {
      typedef NoEventSetupInit<T> type;
    };
  }
}

#endif
