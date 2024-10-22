// system include files
#include <memory>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRelValAnalyzer.h"

using std::cout;
using std::endl;
using std::string;

//
// constructors and destructor
//
L1RCTRelValAnalyzer::L1RCTRelValAnalyzer(const edm::ParameterSet &iConfig)
    : m_rctEmCands(consumes(iConfig.getParameter<edm::InputTag>("rctEmCandsLabel"))),
      m_rctRegions(consumes(iConfig.getParameter<edm::InputTag>("rctRegionsLabel"))) {
  // now do what ever initialization is needed

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  h_emRank = fs->make<TH1F>("emRank", "emRank", 64, 0., 64.);
  h_emIeta = fs->make<TH1F>("emOccupancyIeta", "emOccupancyIeta", 22, 0., 22.);
  h_emIphi = fs->make<TH1F>("emOccupancyIphi", "emOccupancyIphi", 18, 0., 18.);
  h_emIsoOccIetaIphi = fs->make<TH2F>("emIsoOccupancy2D", "emIsoOccupancy2D", 22, 0., 22., 18, 0., 18.);
  h_emNonIsoOccIetaIphi = fs->make<TH2F>("emNonIsoOccupancy2D", "emNonIsoOccupancy2D", 22, 0., 22., 18, 0., 18.);

  h_regionSum = fs->make<TH1F>("regionSum", "regionSum", 100, 0., 100.);
  h_regionSumIetaIphi = fs->make<TH2F>("regionSumEtWeighted2D", "regionSumEtWeighted2D", 22, 0., 22., 18, 0., 18.);
  h_regionOccIetaIphi = fs->make<TH2F>("regionOccupancy2D", "regionOccupancy2D", 22, 0., 22., 18, 0., 18.);
}

L1RCTRelValAnalyzer::~L1RCTRelValAnalyzer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1RCTRelValAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // as in L1GctTestAnalyzer.cc
  Handle<L1CaloEmCollection> rctEmCands = iEvent.getHandle(m_rctEmCands);
  Handle<L1CaloRegionCollection> rctRegions = iEvent.getHandle(m_rctRegions);

  for (auto em = rctEmCands->begin(); em != rctEmCands->end(); em++) {
    if ((*em).rank() > 0) {
      h_emRank->Fill((*em).rank());
      h_emIeta->Fill((*em).regionId().ieta());
      h_emIphi->Fill((*em).regionId().iphi());
      if ((*em).isolated()) {
        h_emIsoOccIetaIphi->Fill((*em).regionId().ieta(), (*em).regionId().iphi());
      } else {
        h_emNonIsoOccIetaIphi->Fill((*em).regionId().ieta(), (*em).regionId().iphi());
      }
    }
  }

  for (auto rgn = rctRegions->begin(); rgn != rctRegions->end(); rgn++) {
    if ((*rgn).et() > 0) {
      h_regionSum->Fill((*rgn).et());
      h_regionSumIetaIphi->Fill((*rgn).gctEta(), (*rgn).gctPhi(), (*rgn).et());
      h_regionOccIetaIphi->Fill((*rgn).gctEta(), (*rgn).gctPhi());
    }
  }
}
