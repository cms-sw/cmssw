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
  bool readMultiBX_;

  TTree* tree_;
  int b_ge0, b_ge1, b_ge2;
};

GEMPackingTester::GEMPackingTester(const edm::ParameterSet& iConfig)
    : fedToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fed"))),
      gemDigiToken_(consumes<GEMDigiCollection>(iConfig.getParameter<edm::InputTag>("gemDigi"))),
      gemSimDigiToken_(consumes<GEMDigiCollection>(iConfig.getParameter<edm::InputTag>("gemSimDigi"))),
      readMultiBX_(iConfig.getParameter<bool>("readMultiBX")) {
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

  auto const& fed_buffers = iEvent.get(fedToken_);

  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXGEMFEDID; ++fedId) {
    const FEDRawData& fedData = fed_buffers.FEDData(fedId);

    if (fedId == 1473 or fedId == 1474)
      b_ge0 += fedData.size();
    if (fedId == 1467 or fedId == 1468)
      b_ge1 += fedData.size();
    if (fedId == 1469 or fedId == 1470)
      b_ge2 += fedData.size();
  }

  auto const& gemDigis = iEvent.get(gemDigiToken_);
  auto const& gemSimDigis = iEvent.get(gemSimDigiToken_);

  for (auto const& simDigi : gemSimDigis) {
    const GEMDetId& gemId = simDigi.first;
    const GEMDigiCollection::Range& sim = simDigi.second;
    const GEMDigiCollection::Range& packed = gemDigis.get(gemId);

    for (auto digi = sim.first; digi != sim.second; ++digi) {
      if (!readMultiBX_ && digi->bx() != 0)
        continue;

      bool foundDigi = false;
      for (auto unpackeddigi = packed.first; unpackeddigi != packed.second; ++unpackeddigi) {
        if ((digi->strip() == unpackeddigi->strip()) && (digi->bx() == unpackeddigi->bx()))
          foundDigi = true;
      }
      if (!foundDigi) {
        edm::LogInfo("GEMPackingTester") << "simMuonGEMDigi NOT found " << gemId << " " << digi->strip() << " "
                                         << digi->bx();
        for (auto unpackeddigi = packed.first; unpackeddigi != packed.second; ++unpackeddigi) {
          edm::LogInfo("GEMPackingTester") << "rec " << unpackeddigi->strip() << " " << unpackeddigi->bx();
        }
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
  desc.add<bool>("readMultiBX", false);
  descriptions.add("GEMPackingTester", desc);
}

DEFINE_FWK_MODULE(GEMPackingTester);
