

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>

class EcalSeverityLevelAlgoAnalyzer : public edm::EDAnalyzer {
public:
  EcalSeverityLevelAlgoAnalyzer(const edm::ParameterSet& ps);

protected:
  void analyze(edm::Event const& iEvent, const edm::EventSetup& iSetup) override;

private:
  TH1F* shisto_;
  TH1F* fhisto_;
};

DEFINE_FWK_MODULE(EcalSeverityLevelAlgoAnalyzer);

EcalSeverityLevelAlgoAnalyzer::EcalSeverityLevelAlgoAnalyzer(const edm::ParameterSet& ps) {
  edm::Service<TFileService> fs;
  shisto_ = fs->make<TH1F>("SeverityLevel", "SeverityLevel", 6, 0, 6);
  fhisto_ = fs->make<TH1F>("Flags", "Flags", 32, 0, 32);
}

void EcalSeverityLevelAlgoAnalyzer::analyze(edm::Event const& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<EcalRecHitCollection> rechits;
  iEvent.getByLabel("ecalRecHit", "EcalRecHitsEB", rechits);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  EcalRecHitCollection::const_iterator rechit = rechits->begin();
  for (; rechit != rechits->end(); ++rechit) {
    for (int flag = 0; flag < 32; ++flag) {
      if (rechit->checkFlag(flag))
        fhisto_->Fill(flag);
    }

    int severity = sevlv->severityLevel(rechit->id(), *rechits);
    shisto_->Fill(severity);
  }
}
