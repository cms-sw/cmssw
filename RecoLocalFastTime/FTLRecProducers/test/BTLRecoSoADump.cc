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
#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitHostCollection.h"

class BTLRecoSoADump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BTLRecoSoADump(const edm::ParameterSet&);
  ~BTLRecoSoADump() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<FTLUncalibratedRecHitCollection> tok_BTL_uncreco;
  edm::EDGetTokenT<FTLRecHitCollection> tok_BTL_reco;
  edm::EDGetTokenT<btlrechit::BTLBaseRecHitHostCollection> tok_BTL_basereco_SoA;
  edm::EDGetTokenT<btlrechit::BTLRecHitHostCollection> tok_BTL_reco_SoA;
};

BTLRecoSoADump::BTLRecoSoADump(const edm::ParameterSet& iConfig)

{
  tok_BTL_uncreco = consumes<FTLUncalibratedRecHitCollection>(edm::InputTag("mtdUncalibratedRecHits", "FTLBarrel"));
  tok_BTL_reco = consumes<FTLRecHitCollection>(edm::InputTag("mtdRecHits", "FTLBarrel"));
  tok_BTL_basereco_SoA = consumes<btlrechit::BTLBaseRecHitHostCollection>(edm::InputTag("btlBaseRecHitsSoA"));
  tok_BTL_reco_SoA = consumes<btlrechit::BTLRecHitHostCollection>(edm::InputTag("btlRecHitsSoA"));
}

BTLRecoSoADump::~BTLRecoSoADump() {}

//
// member functions
//

// ------------ method called for each event ------------
void BTLRecoSoADump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;

  edm::Handle<FTLUncalibratedRecHitCollection> h_BTL_uncreco;
  iEvent.getByToken(tok_BTL_uncreco, h_BTL_uncreco);

  edm::Handle<FTLRecHitCollection> h_BTL_reco;
  iEvent.getByToken(tok_BTL_reco, h_BTL_reco);

  edm::Handle<btlrechit::BTLBaseRecHitHostCollection> h_BTL_basereco_SoA;
  iEvent.getByToken(tok_BTL_basereco_SoA, h_BTL_basereco_SoA);

  edm::Handle<btlrechit::BTLRecHitHostCollection> h_BTL_reco_SoA;
  iEvent.getByToken(tok_BTL_reco_SoA, h_BTL_reco_SoA);

  // --- BTL Uncalib  RECOs:
  std::cout << " ----------------------------------------" << std::endl;
  if (!h_BTL_uncreco->empty()) {
    std::cout << " BTL Uncalib RECO collection:" << std::endl;

    for (const auto& recHit : *h_BTL_uncreco) {
      MTDDetId mtdDetId(recHit.id());
      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() << "  subdet = " << mtdDetId.mtdSubDetector()
                << "  rawID = " << mtdDetId.rawId() << std::endl;

      std::cout << "  amplitude = " << recHit.amplitude().first << "  time = " << recHit.time().first
                << "  amplitude = " << recHit.amplitude().second << "  time = " << recHit.time().second
                << "  position = " << recHit.position() << "  time error = " << recHit.timeError() << std::endl;

    }  // recHit loop
  }

  if (h_BTL_basereco_SoA->view().metadata().size() > 0) {
    std::cout << " BTL Base RECO SoA collection: " << h_BTL_basereco_SoA->view().metadata().size() << "\n" << std::endl;

    for (int i = 0; i < h_BTL_basereco_SoA->view().metadata().size(); i++) {
      std::cout << h_BTL_basereco_SoA->view()[i] << "\n" << std::endl;
    }

  }  // if ( h_BTL_reco_soa->size() > 0 )

  // --- BTL RECOs:

  std::cout << " ----------------------------------------" << std::endl;
  if (!h_BTL_reco->empty()) {
    std::cout << " BTL RECO collection:" << std::endl;

    for (const auto& recHit : *h_BTL_reco) {
      MTDDetId mtdDetId(recHit.id());

      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() << "  subdet = " << mtdDetId.mtdSubDetector()
                << "  rawID = " << mtdDetId.rawId() << std::endl;

      std::cout << "       energy = " << recHit.energy() << "  time = " << recHit.time()
                << "  position = " << recHit.position() << " position error = " << recHit.positionError()
                << "  time error = " << recHit.timeError() << std::endl;

    }  // recHit loop

  }  // if ( h_BTL_reco->size() > 0 )

  if (h_BTL_reco_SoA->view().metadata().size() > 0) {
    std::cout << " BTL RECO SoA collection: " << h_BTL_reco_SoA->view().metadata().size() << "\n" << std::endl;
    for (int i = 0; i < h_BTL_reco_SoA->view().metadata().size(); i++) {
      std::cout << h_BTL_reco_SoA->view()[i] << "\n" << std::endl;
    }

    //  MTDDetId mtdDetId(h_BTL_reco_SoA->view()[i].detId());
  }  // if ( h_BTL_reco_soa->size() > 0 )
}

// ------------ method called once each job just before starting event loop  ------------
void BTLRecoSoADump::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void BTLRecoSoADump::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BTLRecoSoADump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTLRecoSoADump);
