// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "EgammaAnalysis/ElectronTools/interface/EcalIsolationCorrector.h"

#include "TH1F.h"

#include <iostream>

class CorrectECALIsolation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CorrectECALIsolation(const edm::ParameterSet&);
  ~CorrectECALIsolation() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ParameterSet conf_;

  bool isData_;
  edm::EDGetTokenT<reco::GsfElectronCollection> tokenGsfElectrons_;

  TH1F* uncorrectedIsolationEB_;
  TH1F* correctedIsolationEB_;
  TH1F* uncorrectedIsolationEE_;
  TH1F* correctedIsolationEE_;
};

CorrectECALIsolation::CorrectECALIsolation(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  usesResource(TFileService::kSharedResource);

  isData_ = iConfig.getUntrackedParameter<bool>("isData", false);
  tokenGsfElectrons_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("Electrons"));

  edm::Service<TFileService> fs;
  uncorrectedIsolationEB_ = fs->make<TH1F>("uncorrectedIsolationEB", "uncorrected IsolationEB", 50, 0, 10);
  correctedIsolationEB_ = fs->make<TH1F>("correctedIsolationEB", "corrected IsolationEB", 50, 0, 10);
  uncorrectedIsolationEE_ = fs->make<TH1F>("uncorrectedIsolationEE", "uncorrected IsolationEE", 50, 0, 10);
  correctedIsolationEE_ = fs->make<TH1F>("correctedIsolationEE", "corrected IsolationEE", 50, 0, 10);
}

CorrectECALIsolation::~CorrectECALIsolation() {}

void CorrectECALIsolation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::GsfElectronCollection> theEGammaCollection;
  iEvent.getByToken(tokenGsfElectrons_, theEGammaCollection);
  const reco::GsfElectronCollection theEGamma = *(theEGammaCollection.product());

  // Setup a corrector for electrons
  EcalIsolationCorrector ecalIsoCorr(true);

  unsigned nele = theEGammaCollection->size();

  for (unsigned iele = 0; iele < nele; ++iele) {
    reco::GsfElectronRef myElectronRef(theEGammaCollection, iele);

    float uncorrIso = myElectronRef->dr03EcalRecHitSumEt();
    float corrIso = ecalIsoCorr.correctForHLTDefinition(*myElectronRef, iEvent.id().run(), isData_);
    std::cout << "Uncorrected Isolation Sum: " << uncorrIso << " - Corrected: " << corrIso << std::endl;

    if (myElectronRef->isEB()) {
      uncorrectedIsolationEB_->Fill(uncorrIso);
      correctedIsolationEB_->Fill(corrIso);
    } else {
      uncorrectedIsolationEE_->Fill(uncorrIso);
      correctedIsolationEE_->Fill(corrIso);
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CorrectECALIsolation);
