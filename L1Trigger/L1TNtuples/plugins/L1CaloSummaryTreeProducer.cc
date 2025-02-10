// Producer for the CaloSummary cards emulator
// Author: Andrew Loeliger
// Presumably the L1TNtuple Format is going to last very long with the advent of L1NanoAOD,
// But as of now, this is the only way to get CICADA into official menu studying tools

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisCaloSummaryDataFormat.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/CICADA.h"

class L1CaloSummaryTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1CaloSummaryTreeProducer(const edm::ParameterSet&);
  ~L1CaloSummaryTreeProducer() override;

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

public:
  L1Analysis::L1AnalysisCaloSummaryDataFormat* caloSummaryData_;

private:
  const edm::EDGetTokenT<l1t::CICADABxCollection> scoreToken_;
  const edm::EDGetTokenT<L1CaloRegionCollection> regionToken_;
  edm::Service<TFileService> fs_;
  TTree* tree_;
};

L1CaloSummaryTreeProducer::L1CaloSummaryTreeProducer(const edm::ParameterSet& iConfig)
    : scoreToken_(consumes<l1t::CICADABxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("scoreToken"))),
      regionToken_(consumes<L1CaloRegionCollection>(iConfig.getUntrackedParameter<edm::InputTag>("regionToken"))) {
  usesResource(TFileService::kSharedResource);
  tree_ = fs_->make<TTree>("L1CaloSummaryTree", "L1CaloSummaryTree");
  tree_->Branch("CaloSummary", "L1Analysis::L1AnalysisCaloSummaryDataFormat", &caloSummaryData_, 32000, 3);

  caloSummaryData_ = new L1Analysis::L1AnalysisCaloSummaryDataFormat();
}

void L1CaloSummaryTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  caloSummaryData_->Reset();

  edm::Handle<L1CaloRegionCollection> regions;
  iEvent.getByToken(regionToken_, regions);

  if (regions.isValid()) {
    for (const auto& itr : *regions) {
      caloSummaryData_->modelInput[itr.gctPhi()][itr.gctEta() - 4] =
          itr.et();  //4 is subtracted off of the Eta to account for the 4+4 forward/backward HF regions that are not used in CICADA. These take offset the iEta by 4
    }
  } else {
    edm::LogWarning("L1Ntuple") << "Could not find region regions. CICADA model input will not be filled";
  }

  edm::Handle<l1t::CICADABxCollection> score;
  iEvent.getByToken(scoreToken_, score);
  if (score.isValid())
    caloSummaryData_->CICADAScore = score->at(0, 0);
  else
    edm::LogWarning("L1Ntuple") << "Could not find a proper CICADA score. CICADA score will not be filled.";

  tree_->Fill();
}

L1CaloSummaryTreeProducer::~L1CaloSummaryTreeProducer() { delete caloSummaryData_; }

DEFINE_FWK_MODULE(L1CaloSummaryTreeProducer);
