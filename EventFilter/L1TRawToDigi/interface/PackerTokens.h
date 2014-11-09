#ifndef PackerTokens_h
#define PackerTokens_h

namespace edm {
   class ConsumesCollector;
   class ParameterSet;
}

namespace l1t {
   class PackerTokens {
      public:
         PackerTokens(const edm::ParameterSet&, edm::ConsumesCollector&) {};
   };
}

#endif
