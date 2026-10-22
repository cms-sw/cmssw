#include <memory>
#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitHostCollection.h"

class ETLRecoSoADump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ETLRecoSoADump(const edm::ParameterSet&);
  ~ETLRecoSoADump() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<FTLUncalibratedRecHitCollection> tok_ETL_uncreco;
  edm::EDGetTokenT<FTLRecHitCollection> tok_ETL_reco;
  edm::EDGetTokenT<etlrechit::ETLBaseRecHitHostCollection> tok_ETL_basereco_SoA;
  edm::EDGetTokenT<etlrechit::ETLRecHitHostCollection> tok_ETL_reco_SoA;
};

ETLRecoSoADump::ETLRecoSoADump(const edm::ParameterSet& iConfig)

{
  tok_ETL_uncreco = consumes<FTLUncalibratedRecHitCollection>(edm::InputTag("mtdUncalibratedRecHits", "FTLEndcap"));
  tok_ETL_reco = consumes<FTLRecHitCollection>(edm::InputTag("mtdRecHits", "FTLEndcap"));
  tok_ETL_basereco_SoA = consumes<etlrechit::ETLBaseRecHitHostCollection>(edm::InputTag("etlBaseRecHitsSoA"));
  tok_ETL_reco_SoA = consumes<etlrechit::ETLRecHitHostCollection>(edm::InputTag("etlRecHitsSoA"));
}

ETLRecoSoADump::~ETLRecoSoADump() {}

//
// member functions
//

// ------------ method called for each event ------------
void ETLRecoSoADump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;

  edm::Handle<FTLUncalibratedRecHitCollection> h_ETL_uncreco;
  iEvent.getByToken(tok_ETL_uncreco, h_ETL_uncreco);

  edm::Handle<FTLRecHitCollection> h_ETL_reco;
  iEvent.getByToken(tok_ETL_reco, h_ETL_reco);

  edm::Handle<etlrechit::ETLBaseRecHitHostCollection> h_ETL_basereco_SoA;
  iEvent.getByToken(tok_ETL_basereco_SoA, h_ETL_basereco_SoA);

  edm::Handle<etlrechit::ETLRecHitHostCollection> h_ETL_reco_SoA;
  iEvent.getByToken(tok_ETL_reco_SoA, h_ETL_reco_SoA);

  // --- ETL Uncalib  RECOs:
  std::cout << " ----------------------------------------" << std::endl;
  if (!h_ETL_uncreco->empty()) {
    std::cout << " ETL Uncalib RECO collection:" << std::endl;

    for (const auto& recHit : *h_ETL_uncreco) {
      MTDDetId mtdDetId(recHit.id());
      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() << "  subdet = " << mtdDetId.mtdSubDetector()
                << "  rawID = " << mtdDetId.rawId() << std::endl;

      std::cout << "  amplitude = " << recHit.amplitude().first << "  time = " << recHit.time().first
                << "  amplitude = " << recHit.amplitude().second << "  time = " << recHit.time().second
                << "  position = " << recHit.position() << "  time error = " << recHit.timeError() << std::endl;

    }  // recHit loop
  }

  if (h_ETL_basereco_SoA->view().metadata().size() > 0) {
    std::cout << " ETL Base RECO SoA collection: " << h_ETL_basereco_SoA->view().metadata().size() << "\n" << std::endl;

    for (int i = 0; i < h_ETL_basereco_SoA->view().metadata().size(); i++) {
      std::cout << h_ETL_basereco_SoA->view()[i] << "\n" << std::endl;
    }

  }  // if ( h_ETL_reco_soa->size() > 0 )

  // --- ETL RECOs:

  std::cout << " ----------------------------------------" << std::endl;
  if (!h_ETL_reco->empty()) {
    std::cout << " ETL RECO collection:" << std::endl;

    for (const auto& recHit : *h_ETL_reco) {
      MTDDetId mtdDetId(recHit.id());

      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() << "  subdet = " << mtdDetId.mtdSubDetector()
                << "  rawID = " << mtdDetId.rawId() << std::endl;

      std::cout << "       energy = " << recHit.energy() << "  time = " << recHit.time()
                << "  position = " << recHit.position() << " position error = " << recHit.positionError()
                << "  time error = " << recHit.timeError() << std::endl;

    }  // recHit loop

  }  // if ( h_ETL_reco->size() > 0 )

  if (h_ETL_reco_SoA->view().metadata().size() > 0) {
    std::cout << " ETL RECO SoA collection: " << h_ETL_reco_SoA->view().metadata().size() << "\n" << std::endl;
    for (int i = 0; i < h_ETL_reco_SoA->view().metadata().size(); i++) {
      std::cout << h_ETL_reco_SoA->view()[i] << "\n" << std::endl;
    }

    //  MTDDetId mtdDetId(h_ETL_reco_SoA->view()[i].detId());
  }  // if ( h_ETL_reco_soa->size() > 0 )
}

// ------------ method called once each job just before starting event loop  ------------
void ETLRecoSoADump::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void ETLRecoSoADump::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ETLRecoSoADump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ETLRecoSoADump);
