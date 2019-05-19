#ifndef L1ComparatorRun2_h
#define L1ComparatorRun2_h

#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

//#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/EDProducer.h"

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

  class L1ComparatorRun2 : public edm::EDProducer {
  public:
    explicit L1ComparatorRun2(const edm::ParameterSet& ps);
    ~L1ComparatorRun2() override;

  private:
    void produce(edm::Event&, edm::EventSetup const&) override;

    edm::EDGetToken JetDataToken_;
    edm::EDGetToken JetEmulToken_;
    edm::EDGetToken EGammaDataToken_;
    edm::EDGetToken EGammaEmulToken_;
    edm::EDGetToken TauDataToken_;
    edm::EDGetToken TauEmulToken_;
    edm::EDGetToken EtSumDataToken_;
    edm::EDGetToken EtSumEmulToken_;
    edm::EDGetToken CaloTowerDataToken_;
    edm::EDGetToken CaloTowerEmulToken_;

    int bxMax_;
    int bxMin_;

    bool doLayer2_;
    bool doLayer1_;
  };
};  // namespace l1t

#endif
