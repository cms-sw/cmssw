#ifndef L1THIonImp_H
#define L1THIonImp_H

// user include files

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class L1THIonImp : public DQMEDAnalyzer {
public:
  // Constructor
  L1THIonImp(const edm::ParameterSet& ps);

  // Destructor
  ~L1THIonImp() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  //virtual std::vector<int> SortMinBiasBit(std::vector<int>, std::vector<int>);
  virtual std::vector<int> SortMinBiasBit(uint16_t, uint16_t);

private:
  // ----------member data ---------------------------

  // Jet and EM stuff
  MonitorElement* l1GctCenJetsEtEtaPhi_;
  MonitorElement* l1GctForJetsEtEtaPhi_;
  MonitorElement* l1GctTauJetsEtEtaPhi_;
  MonitorElement* l1GctIsoEmRankEtaPhi_;
  MonitorElement* l1GctNonIsoEmRankEtaPhi_;

  MonitorElement* l1GctCenJetsOccEtaPhi_;
  MonitorElement* l1GctForJetsOccEtaPhi_;
  MonitorElement* l1GctTauJetsOccEtaPhi_;
  MonitorElement* l1GctIsoEmOccEtaPhi_;
  MonitorElement* l1GctNonIsoEmOccEtaPhi_;

  MonitorElement* l1GctCenJetsRank_;
  MonitorElement* l1GctForJetsRank_;
  MonitorElement* l1GctTauJetsRank_;
  MonitorElement* l1GctIsoEmRank_;
  MonitorElement* l1GctNonIsoEmRank_;

  MonitorElement* l1GctAllJetsOccRankBx_;
  MonitorElement* l1GctAllEmOccRankBx_;

  // Energy sums stuff
  MonitorElement* l1GctEtMiss_;
  MonitorElement* l1GctEtMissPhi_;
  MonitorElement* l1GctEtMissOf_;
  MonitorElement* l1GctEtMissOccBx_;
  MonitorElement* l1GctEtTotal_;
  MonitorElement* l1GctEtTotalOf_;
  MonitorElement* l1GctEtTotalOccBx_;
  MonitorElement* l1GctEtHad_;
  MonitorElement* l1GctEtHadOf_;
  MonitorElement* l1GctEtHadOccBx_;
  MonitorElement* l1GctEtTotalEtHadCorr_;
  MonitorElement* l1GctHFRingETSum_;
  MonitorElement* l1GctHFRingETDiff_;

  // HF Rings stuff
  MonitorElement* l1GctHFRing1PosEtaNegEta_;
  MonitorElement* l1GctHFRing1TowerCountPosEtaNegEta_;
  MonitorElement* l1GctHFRing2TowerCountPosEtaNegEta_;
  MonitorElement* l1GctHFRing1TowerCountPosEta_;
  MonitorElement* l1GctHFRing1TowerCountNegEta_;
  MonitorElement* l1GctHFRing2TowerCountPosEta_;
  MonitorElement* l1GctHFRing2TowerCountNegEta_;
  MonitorElement* l1GctHFRing1ETSumPosEta_;
  MonitorElement* l1GctHFRing1ETSumNegEta_;
  MonitorElement* l1GctHFRingRatioPosEta_;
  MonitorElement* l1GctHFRingETSumOccBx_;
  MonitorElement* l1GctHFRingTowerCountOccBx_;
  MonitorElement* l1GctMinBiasBitHFEt_;

  MonitorElement* HFNegEnergy_;
  MonitorElement* HFPosEnergy_;
  MonitorElement* HFEnergy_;

  MonitorElement* DECorr_[3][8];

  MonitorElement* centralityCorr_;
  MonitorElement* centralityExtCorr_;
  MonitorElement* MinBiasCorr_;

  edm::InputTag gctCenJetsDataSource_;
  edm::InputTag gctForJetsDataSource_;
  edm::InputTag gctTauJetsDataSource_;
  edm::InputTag gctIsoTauJetsDataSource_;
  edm::InputTag gctEnergySumsDataSource_;
  edm::InputTag gctIsoEmDataSource_;
  edm::InputTag gctNonIsoEmDataSource_;

  edm::EDGetTokenT<L1CaloRegionCollection> rctSource_L1CRCollection_;

  edm::InputTag gctCenJetsEmulSource_;
  edm::InputTag gctForJetsEmulSource_;
  edm::InputTag gctTauJetsEmulSource_;
  edm::InputTag gctIsoTauJetsEmulSource_;
  edm::InputTag gctEnergySumsEmulSource_;
  edm::InputTag gctIsoEmEmulSource_;
  edm::InputTag gctNonIsoEmEmulSource_;

  //define Token(-s)
  edm::EDGetTokenT<L1GctEmCandCollection> gctIsoEmSourceDataToken_;
  edm::EDGetTokenT<L1GctEmCandCollection> gctNonIsoEmSourceDataToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctCenJetsSourceDataToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctForJetsSourceDataToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctTauJetsSourceDataToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctIsoTauJetsSourceDataToken_;
  edm::EDGetTokenT<L1GctHFRingEtSumsCollection> gctEnergySumsSourceDataToken_;
  edm::EDGetTokenT<L1GctHFBitCountsCollection> l1HFCountsDataToken_;
  edm::EDGetTokenT<L1GctEtMissCollection> l1EtMissDataToken_;
  edm::EDGetTokenT<L1GctHtMissCollection> l1HtMissDataToken_;
  edm::EDGetTokenT<L1GctEtHadCollection> l1EtHadDataToken_;
  edm::EDGetTokenT<L1GctEtTotalCollection> l1EtTotalDataToken_;

  edm::EDGetTokenT<L1GctEmCandCollection> gctIsoEmSourceEmulToken_;
  edm::EDGetTokenT<L1GctEmCandCollection> gctNonIsoEmSourceEmulToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctCenJetsSourceEmulToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctForJetsSourceEmulToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctTauJetsSourceEmulToken_;
  edm::EDGetTokenT<L1GctJetCandCollection> gctIsoTauJetsSourceEmulToken_;
  edm::EDGetTokenT<L1GctHFRingEtSumsCollection> gctEnergySumsSourceEmulToken_;
  edm::EDGetTokenT<L1GctHFBitCountsCollection> l1HFCountsEmulToken_;
  edm::EDGetTokenT<L1GctEtMissCollection> l1EtMissEmulToken_;
  edm::EDGetTokenT<L1GctHtMissCollection> l1HtMissEmulToken_;
  edm::EDGetTokenT<L1GctEtHadCollection> l1EtHadEmulToken_;
  edm::EDGetTokenT<L1GctEtTotalCollection> l1EtTotalEmulToken_;
};

#endif
