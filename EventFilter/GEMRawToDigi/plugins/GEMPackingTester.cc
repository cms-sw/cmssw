#include <memory>
#include <iostream>
#include <TTree.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

using namespace std;
class GEMPackingTester : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit GEMPackingTester(const edm::ParameterSet&);
  ~GEMPackingTester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<FEDRawDataCollection> fedToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemSimDigiToken_;

  TTree* tree_;
  int b_ge0, b_ge1, b_ge2;
};

GEMPackingTester::GEMPackingTester(const edm::ParameterSet& iConfig)
    : fedToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fed"))),
      gemDigiToken_(consumes<GEMDigiCollection>(iConfig.getParameter<edm::InputTag>("gemDigi"))),
      gemSimDigiToken_(consumes<GEMDigiCollection>(iConfig.getParameter<edm::InputTag>("gemSimDigi"))) {
  usesResource("TFileService");
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("fed", "fed");
  tree_->Branch("ge0", &b_ge0, "ge0/I");
  tree_->Branch("ge1", &b_ge1, "ge1/I");
  tree_->Branch("ge2", &b_ge2, "ge2/I");
}

GEMPackingTester::~GEMPackingTester() {}

void GEMPackingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  b_ge0 = 0;
  b_ge1 = 0;
  b_ge2 = 0;

  edm::Handle<FEDRawDataCollection> fed_buffers;
  iEvent.getByToken(fedToken_, fed_buffers);
  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXGEMFEDID; ++fedId) {
    const FEDRawData& fedData = fed_buffers->FEDData(fedId);

    //int nWords = fedData.size() / sizeof(uint64_t);
    // std::cout << " fedId:" << fedId
    //           << " fed Size:" << fedData.size()
    //           << " words:" << nWords
    //           << std::endl;

    if (fedId == 1473 or fedId == 1474)
      b_ge0 += fedData.size();
    if (fedId == 1467 or fedId == 1468)
      b_ge1 += fedData.size();
    if (fedId == 1469 or fedId == 1470)
      b_ge2 += fedData.size();
  }
  // std::cout << " ge0:" << b_ge0
  //           << " ge1:" << b_ge1
  //           << " ge2:" << b_ge2
  //           << std::endl;

  edm::Handle<GEMDigiCollection> gemDigis;
  iEvent.getByToken(gemDigiToken_, gemDigis);
  edm::Handle<GEMDigiCollection> gemSimDigis;
  iEvent.getByToken(gemSimDigiToken_, gemSimDigis);

  for (auto const& simDigi : *gemSimDigis) {
    const GEMDetId& gemId = simDigi.first;
    const GEMDigiCollection::Range& sim = simDigi.second;
    const GEMDigiCollection::Range& packed = gemDigis->get(gemId);

    int nsims = 0;
    for (auto digi = sim.first; digi != sim.second; ++digi) {
      nsims++;
    }
    int npacked = 0;
    for (auto digi = packed.first; digi != packed.second; ++digi) {
      npacked++;
    }
    if (nsims != npacked) {
      cout << gemId << endl;
      for (auto digi = sim.first; digi != sim.second; ++digi) {
        cout << "sim " << digi->strip() << " " << digi->bx() << endl;
      }
      for (auto digi = packed.first; digi != packed.second; ++digi) {
        cout << "rec " << digi->strip() << " " << digi->bx() << endl;
      }
    }
  }

  tree_->Fill();
}

void GEMPackingTester::beginJob() {}

void GEMPackingTester::endJob() {}

void GEMPackingTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("fed", edm::InputTag("rawDataCollector"));
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("muonGEMDigis"));
  desc.add<edm::InputTag>("gemSimDigi", edm::InputTag("simMuonGEMDigis"));
  descriptions.add("GEMPackingTester", desc);
}

DEFINE_FWK_MODULE(GEMPackingTester);
