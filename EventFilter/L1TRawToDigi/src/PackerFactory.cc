#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

#include "implementations/CaloTowerPacker.h"
#include "implementations/JetPacker.h"

namespace l1t {
   std::vector<PackerFactory*> PackerFactory::factories_ = PackerFactory::createFactories();

   std::vector<PackerFactory*> PackerFactory::createFactories()
   {
      std::vector<PackerFactory*> res;
      res.push_back(new CaloTowerPackerFactory());
      res.push_back(new JetPackerFactory());
      return res;
   }

   PackerList
   PackerFactory::createPackers(const edm::ParameterSet& cfg, unsigned fw, const int fedid)
   {
      PackerList res;
      for (const auto& f: factories_) {
         auto ps = f->create(cfg, fw, fedid);
         res.insert(res.end(), ps.begin(), ps.end());
      }
      return res;
   }
}
