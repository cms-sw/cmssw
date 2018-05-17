#include <memory>
#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class MTDRecoDump : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit MTDRecoDump(const edm::ParameterSet&);
      ~MTDRecoDump();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

  // ----------member data ---------------------------

      edm::EDGetTokenT<FTLRecHitCollection> tok_BTL_reco; 
      edm::EDGetTokenT<FTLRecHitCollection> tok_ETL_reco; 

};


MTDRecoDump::MTDRecoDump(const edm::ParameterSet& iConfig)

{

  tok_BTL_reco = consumes<FTLRecHitCollection>(edm::InputTag("ftlRecHits","FTLBarrel"));
  tok_ETL_reco = consumes<FTLRecHitCollection>(edm::InputTag("ftlRecHits","FTLEndcap"));

}


MTDRecoDump::~MTDRecoDump() {}


//
// member functions
//

// ------------ method called for each event ------------
void
MTDRecoDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;

  edm::Handle<FTLRecHitCollection> h_BTL_reco;
  iEvent.getByToken( tok_BTL_reco, h_BTL_reco );

  edm::Handle<FTLRecHitCollection> h_ETL_reco;
  iEvent.getByToken( tok_ETL_reco, h_ETL_reco );

  // --- BTL RECOs:

  if ( h_BTL_reco->size() > 0 ) {

    std::cout << " ----------------------------------------" << std::endl;
    std::cout << " BTL RECO collection:" << std::endl;
  
    for (const auto& recHit: *h_BTL_reco) {

      MTDDetId mtdDetId(recHit.id());

      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() 
		<< "  subdet = "  << mtdDetId.mtdSubDetector()
		<< "  rawID = " << mtdDetId.rawId() 
		<< std::endl;

      std::cout << "       energy = " << recHit.energy() 
		<< "  time = " << recHit.time() 
		<< "  time error = " << recHit.timeError() 
		<< std::endl;

    } // recHit loop

  } // if ( h_BTL_reco->size() > 0 )


  // --- ETL RECOs:

  if ( h_ETL_reco->size() > 0 ) {

    std::cout << " ----------------------------------------" << std::endl;
    std::cout << " ETL RECO collection:" << std::endl;
  
    for (const auto& recHit: *h_ETL_reco) {

      MTDDetId mtdDetId(recHit.id());

      // --- detector element ID:
      std::cout << "   det ID:  det = " << mtdDetId.det() 
		<< "  subdet = "  << mtdDetId.mtdSubDetector()
		<< "  rawID = " << mtdDetId.rawId() 
		<< std::endl;

      std::cout << "       energy = " << recHit.energy() 
		<< "  time = " << recHit.time() 
		<< "  time error = " << recHit.timeError() 
		<< std::endl;

    } // recHit loop

  } // if ( h_ETL_reco->size() > 0 )
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
MTDRecoDump::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MTDRecoDump::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MTDRecoDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDRecoDump);
