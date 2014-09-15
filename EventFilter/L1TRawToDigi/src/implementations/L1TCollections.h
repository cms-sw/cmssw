#ifndef L1TCollections_h
#define L1TCollections_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   class L1TCollectionsProduces : public UnpackerCollectionsProduces {
      public:
         L1TCollectionsProduces(edm::one::EDProducerBase&);
   };

   class L1TCollections : public UnpackerCollections {
      public:
         L1TCollections(edm::Event& e) :
            UnpackerCollections(e),
            towers_(new CaloTowerBxCollection()),
            egammas_(new EGammaBxCollection()),
            etsums_(new EtSumBxCollection()),
            jets_(new JetBxCollection()),
            taus_(new TauBxCollection()),
            mp_etsums_(new EtSumBxCollection()),
            mp_jets_(new JetBxCollection()) {};

         virtual ~L1TCollections();

         inline CaloTowerBxCollection* getTowers() { return towers_.get(); };
         inline EGammaBxCollection* getEGammas() { return egammas_.get(); };
         inline EtSumBxCollection* getEtSums() { return etsums_.get(); };
         inline JetBxCollection* getJets() { return jets_.get(); };
         inline TauBxCollection* getTaus() { return taus_.get(); };

         inline EtSumBxCollection* getMPEtSums() { return mp_etsums_.get(); };
         inline JetBxCollection* getMPJets() { return mp_jets_.get(); };

      private:
         std::auto_ptr<CaloTowerBxCollection> towers_;
         std::auto_ptr<EGammaBxCollection> egammas_;
         std::auto_ptr<EtSumBxCollection> etsums_;
         std::auto_ptr<JetBxCollection> jets_;
         std::auto_ptr<TauBxCollection> taus_;

         std::auto_ptr<EtSumBxCollection> mp_etsums_;
         std::auto_ptr<JetBxCollection> mp_jets_;
   };
}

#endif
