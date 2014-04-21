#ifndef UnpackerCollections_h
#define UnpackerCollections_h

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

namespace l1t {
   class L1TRawToDigi;

   class UnpackerCollections {
      public:
         UnpackerCollections(edm::Event& event);
         ~UnpackerCollections();

         inline JetBxCollection * const getJetCollection() const { return jets_.get(); };
         inline TauBxCollection * const getTauCollection() const { return taus_.get(); };
         inline EGammaBxCollection * const getEGammaCollection() const { return egammas_.get(); };
         inline EtSumBxCollection * const getEtSumCollection() const { return etsums_.get(); };
         inline CaloTowerBxCollection * const getCaloTowerCollection() const { return calotowers_.get(); };

         static void registerCollections(L1TRawToDigi*);

      private:
         // Keep this a singular object.
         UnpackerCollections(const UnpackerCollections&);
         UnpackerCollections& operator=(const UnpackerCollections&);

         edm::Event& event_;

         std::auto_ptr<JetBxCollection> jets_;
         std::auto_ptr<TauBxCollection> taus_;
         std::auto_ptr<EGammaBxCollection> egammas_;
         std::auto_ptr<EtSumBxCollection> etsums_;
         std::auto_ptr<CaloTowerBxCollection> calotowers_;
   };
}

#endif
