#include "FWCore/Framework/interface/one/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerProvider.h"

#include "CaloCollections.h"

namespace l1t {
   class CaloSetup : public UnpackerProvider {
      public:
         CaloSetup(edm::one::EDProducerBase& prod) : UnpackerProvider(prod) {
            prod.produces<CaloTowerBxCollection>();
            prod.produces<EGammaBxCollection>();
            prod.produces<EtSumBxCollection>();
            prod.produces<JetBxCollection>();
            prod.produces<TauBxCollection>();

            prod.produces<EtSumBxCollection>("MP");
            prod.produces<JetBxCollection>("MP");
         };

         virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
            return std::unique_ptr<UnpackerCollections>(new CaloCollections(e));
         };

         virtual UnpackerMap getUnpackers() override {
            auto tower_unp = UnpackerFactory::get()->make("CaloTowerUnpacker");
            auto egamma_unp = UnpackerFactory::get()->make("EGammaUnpacker");
            auto etsum_unp = UnpackerFactory::get()->make("EtSumUnpacker");
            auto jet_unp = UnpackerFactory::get()->make("JetUnpacker");
            auto tau_unp = UnpackerFactory::get()->make("TauUnpacker");

            auto mp_unp = UnpackerFactory::get()->make("MPUnpacker");

            UnpackerMap res;
            res[std::make_tuple(1, 1, 1, 1)] = egamma_unp;
            res[std::make_tuple(1, 1, 3, 1)] = etsum_unp;
            res[std::make_tuple(1, 1, 5, 1)] = jet_unp;
            res[std::make_tuple(1, 1, 7, 1)] = tau_unp;

            res[std::make_tuple(2, 1, 1, 1)] = mp_unp;
            res[std::make_tuple(2, 3, 1, 1)] = mp_unp;
            res[std::make_tuple(2, 5, 1, 1)] = mp_unp;
            res[std::make_tuple(2, 7, 1, 1)] = mp_unp;
            res[std::make_tuple(2, 9, 1, 1)] = mp_unp;
            res[std::make_tuple(2, 11, 1, 1)] = mp_unp;

            for (int link = 0; link < 144; link += 2)
               res[std::make_tuple(2, 1, link, 1)] = tower_unp;

            return res;
         };
   };
}

DEFINE_L1T_UNPACKER_PROVIDER(l1t::CaloSetup);
