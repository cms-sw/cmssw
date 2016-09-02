#ifndef CaloCollections_h
#define CaloCollections_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
   namespace stage2 {
     class CaloCollections : public L1TObjectCollections {
         public:
            CaloCollections(edm::Event& e) :
               L1TObjectCollections(e),
               towers_(new CaloTowerBxCollection()),
               egammas_(new EGammaBxCollection()),
               etsums_(new EtSumBxCollection()),
               jets_(new JetBxCollection()),
               taus_(new TauBxCollection()),
               mp_etsums_(new EtSumBxCollection()),
	       mp_jets_(new JetBxCollection()),
		 mp_egammas_(new EGammaBxCollection()),
	       mp_taus_(new TauBxCollection()) {};

            virtual ~CaloCollections();

            inline CaloTowerBxCollection* getTowers() { return towers_.get(); };
            inline EGammaBxCollection* getEGammas() override { return egammas_.get(); };
            inline EtSumBxCollection* getEtSums() override { return etsums_.get(); };
            inline JetBxCollection* getJets() override { return jets_.get(); };
            inline TauBxCollection* getTaus() override { return taus_.get(); };

            inline EtSumBxCollection* getMPEtSums() { return mp_etsums_.get(); };
            inline JetBxCollection* getMPJets() { return mp_jets_.get(); };
	    inline EGammaBxCollection* getMPEGammas() { return mp_egammas_.get(); };
	    inline TauBxCollection* getMPTaus() { return mp_taus_.get(); };

         private:
            std::auto_ptr<CaloTowerBxCollection> towers_;
            std::auto_ptr<EGammaBxCollection> egammas_;
            std::auto_ptr<EtSumBxCollection> etsums_;
            std::auto_ptr<JetBxCollection> jets_;
            std::auto_ptr<TauBxCollection> taus_;

            std::auto_ptr<EtSumBxCollection> mp_etsums_;
            std::auto_ptr<JetBxCollection> mp_jets_;
	    std::auto_ptr<EGammaBxCollection> mp_egammas_;
	    std::auto_ptr<TauBxCollection> mp_taus_;
      };
   }
}

#endif
