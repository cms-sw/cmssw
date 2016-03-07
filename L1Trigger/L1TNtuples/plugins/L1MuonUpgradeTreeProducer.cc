
#include <memory>
// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisUGMT.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisUGMTDataFormat.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSum.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
// output
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"


class L1MuonUpgradeTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1MuonUpgradeTreeProducer(const edm::ParameterSet&);
  ~L1MuonUpgradeTreeProducer();


private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  L1Analysis::L1AnalysisUGMT ugmt;
  L1Analysis::L1AnalysisUGMTDataFormat* ugmtData;
  L1Analysis::L1AnalysisMuTwrDataFormat* twrData;
  L1Analysis::L1AnalysisMuTwrDataFormat* twrRecoData;
  L1Analysis::L1AnalysisMuTwrDataFormat* twr2x2Data;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;
  const float towerEtas[33] = {0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.870,0.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,2.322,2.5,2.650,3.000,3.5,4.0,4.5,5.0};
  // EDM input tags
  edm::InputTag bmtfTag_;
  edm::InputTag omtfTag_;
  edm::InputTag emtfTag_;
  edm::InputTag ugmtTag_;
  edm::InputTag calo2x2Tag_;
  edm::InputTag caloTag_;
  edm::InputTag caloRecoTag_;

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> bmtfToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> omtfToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> emtfToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtToken_;
  edm::EDGetTokenT<CaloTowerCollection> caloRecoToken_;
  edm::EDGetTokenT<l1t::MuonCaloSumBxCollection> caloToken_;
  edm::EDGetTokenT<l1t::MuonCaloSumBxCollection> calo2x2Token_;
};


L1MuonUpgradeTreeProducer::L1MuonUpgradeTreeProducer(const edm::ParameterSet& iConfig) :
  ugmt(),
  bmtfTag_(iConfig.getParameter<edm::InputTag>("bmtfTag")),
  omtfTag_(iConfig.getParameter<edm::InputTag>("omtfTag")),
  emtfTag_(iConfig.getParameter<edm::InputTag>("emtfTag")),
  ugmtTag_(iConfig.getParameter<edm::InputTag>("ugmtTag")),
  calo2x2Tag_(iConfig.getParameter<edm::InputTag>("calo2x2Tag")),
  caloTag_(iConfig.getParameter<edm::InputTag>("caloTag")),
  caloRecoTag_(iConfig.getParameter<edm::InputTag>("caloRecoTag"))
{
  bmtfToken_ = consumes<l1t::RegionalMuonCandBxCollection>(bmtfTag_);
  omtfToken_ = consumes<l1t::RegionalMuonCandBxCollection>(omtfTag_);
  emtfToken_ = consumes<l1t::RegionalMuonCandBxCollection>(emtfTag_);
  caloToken_ = consumes<l1t::MuonCaloSumBxCollection>(caloTag_);
  caloRecoToken_ = mayConsume<CaloTowerCollection>(caloRecoTag_);
  calo2x2Token_ = consumes<l1t::MuonCaloSumBxCollection>(calo2x2Tag_);
  ugmtToken_ = consumes<l1t::MuonBxCollection>(ugmtTag_);

  ugmtData = ugmt.getData();
  twrData = new L1Analysis::L1AnalysisMuTwrDataFormat();
  twr2x2Data = new L1Analysis::L1AnalysisMuTwrDataFormat();
  twrRecoData = new L1Analysis::L1AnalysisMuTwrDataFormat();
  tree_ = fs_->make<TTree>("L1MuonUpgradeTree", "L1MuonUpgradeTree");
  tree_->Branch("L1TMuon", "L1Analysis::L1AnalysisUGMTDataFormat", &ugmtData, 32000, 3);
  tree_->Branch("L1TMuonCalo2x2", "L1Analysis::L1AnalysisMuTwrDataFormat", &twr2x2Data, 32000, 3);
  tree_->Branch("L1TMuonCalo", "L1Analysis::L1AnalysisMuTwrDataFormat", &twrData, 32000, 3);
  tree_->Branch("L1TMuonRecoCalo", "L1Analysis::L1AnalysisMuTwrDataFormat", &twrRecoData, 32000, 3);
}


L1MuonUpgradeTreeProducer::~L1MuonUpgradeTreeProducer()
{
}


void
L1MuonUpgradeTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ugmt.Reset();
  twrData->Reset();
  twr2x2Data->Reset();
  twrRecoData->Reset();

  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfMuons;
  edm::Handle<l1t::RegionalMuonCandBxCollection> emtfMuons;
  edm::Handle<l1t::RegionalMuonCandBxCollection> omtfMuons;
  edm::Handle<l1t::MuonBxCollection> ugmtMuons;
  edm::Handle<l1t::MuonCaloSumBxCollection> calo2x2Twrs;
  edm::Handle<l1t::MuonCaloSumBxCollection> caloTwrs;
  edm::Handle<CaloTowerCollection> caloRecoTwrs;


  iEvent.getByToken(bmtfToken_, bmtfMuons);
  iEvent.getByToken(emtfToken_, emtfMuons);
  iEvent.getByToken(omtfToken_, omtfMuons);
  iEvent.getByToken(ugmtToken_, ugmtMuons);
  iEvent.getByToken(calo2x2Token_, calo2x2Twrs);
  iEvent.getByToken(caloToken_, caloTwrs);
  // iEvent.getByLabel(m_trigTowerTag, trigTowers);
  if (bmtfMuons.isValid() && emtfMuons.isValid() && omtfMuons.isValid() && ugmtMuons.isValid() && calo2x2Twrs.isValid()) {
    ugmt.Set(*ugmtMuons, *bmtfMuons, *omtfMuons, *emtfMuons, false);

    for (int bx = calo2x2Twrs->getFirstBX(); bx <= calo2x2Twrs->getLastBX(); ++bx) {
      for (auto it = calo2x2Twrs->begin(bx); it != calo2x2Twrs->end(bx); ++it) {
        twr2x2Data->packedPt.push_back(it->etBits());
        twr2x2Data->packedEta.push_back(it->hwEta());
        twr2x2Data->packedPhi.push_back(it->hwPhi());
        twr2x2Data->bx.push_back(bx);

        // twrData->pt.push_back(it->etBits()*0.5);
        // int ieta = (it->hwEta() - 27) * 2;
        // float eta = ieta > 0 ? towerEtas[std::abs(ieta)+1] : -towerEtas[std::abs(ieta)+1];
        // twrData->eta.push_back(eta);
        // twrData->phi.push_back((it->hwPhi() * 2 + 1) * 0.087266);
      }
      twr2x2Data->n = calo2x2Twrs->size(bx);
    }

    for (int bx = caloTwrs->getFirstBX(); bx <= caloTwrs->getLastBX(); ++bx) {
      for (auto it = caloTwrs->begin(bx); it != caloTwrs->end(bx); ++it) {
        const l1t::MuonCaloSum& twr = *it;
        twrData->packedPhi.push_back(twr.hwPhi());
        twrData->packedEta.push_back(twr.hwEta());
        twrData->packedPt.push_back(twr.etBits());
        twrData->bx.push_back(bx);
      }
      twrData->n = caloTwrs->size(bx);
    }
  } else {
    edm::LogError("MissingProduct") << "L1Upgrade GMT inputs and / or output not found" << std::endl;
    return;
  }
  if (caloRecoTag_.label() != "none") {
    iEvent.getByToken(caloRecoToken_, caloRecoTwrs);
    for(auto it = caloRecoTwrs->begin(); it != caloRecoTwrs->end(); ++it) {
      twrRecoData->packedPhi.push_back(it->iphi());
      twrRecoData->packedEta.push_back(it->ieta());
      twrRecoData->packedPt.push_back((it->emEt() + it->hadEt())*10);
    }
  }
  tree_->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void
L1MuonUpgradeTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1MuonUpgradeTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MuonUpgradeTreeProducer);
