#ifndef CandAlgos_SetupInitTrait_h
#define CandAlgos_SetupInitTrait_h

namespace edm {
  class EventSetup;
}

namespace reco {
  namespace helpers {
    template<typename Setup>
    struct NoSetupInit {
      static void init( Setup & s, const edm::EventSetup& es ) { }
    };

    template<typename Setup>
    struct SetupInit {
      typedef NoSetupInit<Setup> type;
    };
  }
}

#endif
