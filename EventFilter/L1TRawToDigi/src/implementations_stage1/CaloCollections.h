#ifndef CaloCollections_h
#define CaloCollections_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   namespace stage1 {
      class CaloCollections : public UnpackerCollections {
         public:
            CaloCollections(edm::Event& e) :
               UnpackerCollections(e),
               towers_(new CaloTowerBxCollection()),
               egammas_(new EGammaBxCollection()),
               etsums_(new EtSumBxCollection()),
               jets_(new JetBxCollection()),
               taus_(new TauBxCollection()),
               isotaus_(new TauBxCollection()),
               calospareHFBitCounts_(new CaloSpareBxCollection()),
               calospareHFRingSums_(new CaloSpareBxCollection()) {};

            virtual ~CaloCollections();

            inline CaloTowerBxCollection* getTowers() { return towers_.get(); };
            inline EGammaBxCollection* getEGammas() { return egammas_.get(); };
            inline EtSumBxCollection* getEtSums() { return etsums_.get(); };
            inline JetBxCollection* getJets() { return jets_.get(); };
            inline TauBxCollection* getTaus() { return taus_.get(); };
            inline TauBxCollection* getIsoTaus() { return isotaus_.get(); };
            inline CaloSpareBxCollection* getCaloSpareHFBitCounts() { return calospareHFBitCounts_.get(); };
            inline CaloSpareBxCollection* getCaloSpareHFRingSums() { return calospareHFRingSums_.get(); };
            

         private:
            std::auto_ptr<CaloTowerBxCollection> towers_;
            std::auto_ptr<EGammaBxCollection> egammas_;
            std::auto_ptr<EtSumBxCollection> etsums_;
            std::auto_ptr<JetBxCollection> jets_;
            std::auto_ptr<TauBxCollection> taus_;
            std::auto_ptr<TauBxCollection> isotaus_;
            std::auto_ptr<CaloSpareBxCollection> calospareHFBitCounts_;
            std::auto_ptr<CaloSpareBxCollection> calospareHFRingSums_;
      };
   }
}

#endif
