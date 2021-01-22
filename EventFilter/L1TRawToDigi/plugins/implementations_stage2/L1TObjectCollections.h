#ifndef L1TCollections_h
#define L1TCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
  namespace stage2 {
    class L1TObjectCollections : public UnpackerCollections {
    public:
      L1TObjectCollections(edm::Event& e) : UnpackerCollections(e){};
      ~L1TObjectCollections() override;

      virtual MuonBxCollection* getMuons(const unsigned int copy) { return nullptr; }
      virtual EGammaBxCollection* getEGammas(const unsigned int copy) { return nullptr; }  //= 0;
      virtual EtSumBxCollection* getEtSums(const unsigned int copy) { return nullptr; }
      virtual JetBxCollection* getJets(const unsigned int copy) { return nullptr; }
      virtual TauBxCollection* getTaus(const unsigned int copy) { return nullptr; }

      virtual EcalTrigPrimDigiCollection* getEcalDigisBx(const unsigned int copy) { return nullptr; };
    };
  }  // namespace stage2
}  // namespace l1t

#endif
