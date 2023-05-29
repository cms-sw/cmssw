#ifndef L1ComparatorRun2_h
#define L1ComparatorRun2_h

#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/L1Trigger/interface/CaloSpare.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/L1DataEmulResult.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {

  class L1ComparatorRun2 : public edm::global::EDProducer<> {
  public:
    explicit L1ComparatorRun2(const edm::ParameterSet& ps);
    ~L1ComparatorRun2() override;

  private:
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    edm::EDGetTokenT<JetBxCollection> JetDataToken_;
    edm::EDGetTokenT<JetBxCollection> JetEmulToken_;
    edm::EDGetTokenT<EGammaBxCollection> EGammaDataToken_;
    edm::EDGetTokenT<EGammaBxCollection> EGammaEmulToken_;
    edm::EDGetTokenT<TauBxCollection> TauDataToken_;
    edm::EDGetTokenT<TauBxCollection> TauEmulToken_;
    edm::EDGetTokenT<EtSumBxCollection> EtSumDataToken_;
    edm::EDGetTokenT<EtSumBxCollection> EtSumEmulToken_;
    edm::EDGetTokenT<CaloTowerBxCollection> CaloTowerDataToken_;
    edm::EDGetTokenT<CaloTowerBxCollection> CaloTowerEmulToken_;

    int bxMax_;
    int bxMin_;

    bool doLayer2_;
    bool doLayer1_;
  };
};  // namespace l1t

#endif
