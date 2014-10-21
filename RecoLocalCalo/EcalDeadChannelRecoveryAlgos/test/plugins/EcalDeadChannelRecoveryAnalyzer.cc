#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include <iostream>

class EcalDeadChannelRecoveryAnalyzer: public edm::EDAnalyzer {
 public:
  explicit EcalDeadChannelRecoveryAnalyzer(const edm::ParameterSet&);

 private:
  virtual void beginJob() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

 private:
  edm::EDGetTokenT<EcalRecHitCollection> originalRecHitCollectionT_;
  edm::EDGetTokenT<EcalRecHitCollection> recoveredRecHitCollectionT_;

  TH1D *histoEnergies_;
  TH1D *histoRecovery_;
};

EcalDeadChannelRecoveryAnalyzer::EcalDeadChannelRecoveryAnalyzer(const edm::ParameterSet& iConfig) {
  originalRecHitCollectionT_ =
    consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("originalRecHitCollection"));

  recoveredRecHitCollectionT_ =
    consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recoveredRecHitCollection"));

  std::string titlePrefix = iConfig.getParameter<std::string>("titlePrefix");
  std::string title;

  edm::Service<TFileService> fs;

  title = titlePrefix + "Original energies.";
  histoEnergies_ = fs->make<TH1D>("originalEnergies", title.c_str(), 256, 0 , 10);

  title = titlePrefix + "Recovered energies.";
  histoRecovery_ = fs->make<TH1D>("recoveryEnergies", title.c_str(), 256, 0 , 10);
}

void EcalDeadChannelRecoveryAnalyzer::analyze(const edm::Event& ev,
                                    const edm::EventSetup&) {

  edm::Handle<EcalRecHitCollection> originalRecHits;
  edm::Handle<EcalRecHitCollection> recoveredRecHits;

  ev.getByToken(originalRecHitCollectionT_, originalRecHits);
  ev.getByToken(recoveredRecHitCollectionT_, recoveredRecHits);

  for (auto const& hit: *recoveredRecHits) {
    EcalRecHit original = *(originalRecHits->find(hit.id()));

    if (hit.checkFlag(EcalRecHit::kDead)) {
      // if it didn't change from the original, return
      if (original.checkFlag(EcalRecHit::kDead))
        continue;

      // std::cout << "Failed to recover: " << hit << std::endl;
      // std::cout << "Original: " << original << std::endl;

      histoEnergies_->Fill(original.energy());
    }

    if (hit.checkFlag(EcalRecHit::kNeighboursRecovered)) {
      // if it didn't change from the original, return
      if (original.checkFlag(EcalRecHit::kNeighboursRecovered))
        continue;

      // std::cout << "Recovered: " << hit << std::endl;
      // std::cout << "Original: " << original << std::endl;

      histoEnergies_->Fill(original.energy());
      histoRecovery_->Fill(hit.energy());
    }

        
  }
}

DEFINE_FWK_MODULE(EcalDeadChannelRecoveryAnalyzer);
