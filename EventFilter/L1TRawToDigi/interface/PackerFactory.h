#ifndef PackerFactory_h
#define PackerFactory_h

#include <memory>
#include <vector>

#include "EventFilter/L1TRawToDigi/interface/BasePacker.h"

namespace l1t {
   typedef std::vector<std::shared_ptr<l1t::BasePacker>> PackerList;

   class PackerFactory {
      public:
     static PackerList createPackers(const edm::ParameterSet&, unsigned, const int fedid);

      private:
         virtual PackerList create(const edm::ParameterSet&, unsigned, const int fedid) = 0;

         static std::vector<PackerFactory*> createFactories();
         static std::vector<PackerFactory*> factories_;
   };
}

#endif
