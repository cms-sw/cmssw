#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class TauPackerFactory : public BasePackerFactory {
      public:
         TauPackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual PackerList create(const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
   };
}
