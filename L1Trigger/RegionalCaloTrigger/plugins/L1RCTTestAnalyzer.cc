// system include files
#include <memory>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestAnalyzer.h"

using std::cout;
using std::endl;
using std::string;

//
// constructors and destructor
//
L1RCTTestAnalyzer::L1RCTTestAnalyzer(const edm::ParameterSet &iConfig)
    : showEmCands(iConfig.getUntrackedParameter<bool>("showEmCands")),
      showRegionSums(iConfig.getUntrackedParameter<bool>("showRegionSums")),
      ecalDigisLabel(iConfig.getParameter<edm::InputTag>("ecalDigisLabel")),
      hcalDigisLabel(iConfig.getParameter<edm::InputTag>("hcalDigisLabel")),
      rctDigisLabel(iConfig.getParameter<edm::InputTag>("rctDigisLabel")) {
  // now do what ever initialization is needed

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  emTree = fs->make<TTree>("emTree", "L1 RCT EM tree");
  //   emTree->Branch("emRank",emRank,"emRank/I");
  //   emTree->Branch("emIeta",emIeta,"emIeta/I");
  //   emTree->Branch("emIphi",emIphi,"emIphi/I");
  //   emTree->Branch("emIso" ,emIso ,"emIso/I");
  emTree->Branch("emRank", &emRank);
  emTree->Branch("emIeta", &emIeta);
  emTree->Branch("emIphi", &emIphi);
  emTree->Branch("emIso", &emIso);

  h_emRank = fs->make<TH1F>("emRank", "emRank", 64, 0., 64.);
  h_emRankOutOfTime = fs->make<TH1F>("emRankOutOfTime", "emRankOutOfTime", 64, 0., 64.);
  h_emIeta = fs->make<TH1F>("emIeta", "emIeta", 22, 0., 22.);
  h_emIphi = fs->make<TH1F>("emIphi", "emIphi", 18, 0., 18.);
  h_emIso = fs->make<TH1F>("emIso", "emIso", 2, 0., 2.);
  h_emRankInIetaIphi = fs->make<TH2F>("emRank2D", "emRank2D", 22, 0., 22., 18, 0., 18.);
  h_emIsoInIetaIphi = fs->make<TH2F>("emIso2D", "emIso2D", 22, 0., 22., 18, 0., 18.);
  h_emNonIsoInIetaIphi = fs->make<TH2F>("emNonIso2D", "emNonIso2D", 22, 0., 22., 18, 0., 18.);
  h_emCandTimeSample = fs->make<TH1F>("emCandTimeSample", "emCandTimeSample", 5, -2., 2.);

  h_regionSum = fs->make<TH1F>("regionSum", "regionSum", 100, 0., 100.);
  h_regionIeta = fs->make<TH1F>("regionIeta", "regionIeta", 22, 0., 22.);
  h_regionIphi = fs->make<TH1F>("regionIphi", "regionIphi", 18, 0., 18.);
  h_regionMip = fs->make<TH1F>("regionMip", "regionMipBit", 2, 0., 2.);
  h_regionSumInIetaIphi = fs->make<TH2F>("regionSum2D", "regionSum2D", 22, 0., 22., 18, 0., 18.);
  h_regionFGInIetaIphi = fs->make<TH2F>("regionFG2D", "regionFG2D", 22, 0., 22., 18, 0., 18.);

  h_towerMip = fs->make<TH1F>("towerMip", "towerMipBit", 2, 0., 2.);

  h_ecalTimeSample = fs->make<TH1F>("ecalTimeSample", "ecalTimeSample", 10, 0., 10.);
  h_hcalTimeSample = fs->make<TH1F>("hcalTimeSample", "hcalTimeSample", 10, 0., 10.);

  // get names of modules, producing object collections
}

L1RCTTestAnalyzer::~L1RCTTestAnalyzer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1RCTTestAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example", pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
#endif

  // as in L1GctTestAnalyzer.cc
  Handle<L1CaloEmCollection> rctEmCands;
  Handle<L1CaloRegionCollection> rctRegions;
  Handle<EcalTrigPrimDigiCollection> ecalColl;
  Handle<HcalTrigPrimDigiCollection> hcalColl;

  L1CaloEmCollection::const_iterator em;
  L1CaloRegionCollection::const_iterator rgn;
  EcalTrigPrimDigiCollection::const_iterator ecal;
  HcalTrigPrimDigiCollection::const_iterator hcal;

  iEvent.getByLabel(rctDigisLabel, rctEmCands);
  iEvent.getByLabel(rctDigisLabel, rctRegions);
  iEvent.getByLabel(ecalDigisLabel, ecalColl);
  iEvent.getByLabel(hcalDigisLabel, hcalColl);

  // for sorting later
  L1CaloEmCollection *myL1EmColl = new L1CaloEmCollection;

  for (ecal = ecalColl->begin(); ecal != ecalColl->end(); ecal++) {
    for (unsigned short sample = 0; sample < (*ecal).size(); sample++) {
      h_ecalTimeSample->Fill(sample);
    }
  }

  for (hcal = hcalColl->begin(); hcal != hcalColl->end(); hcal++) {
    h_towerMip->Fill((*hcal).SOI_fineGrain());
    for (unsigned short sample = 0; sample < (*hcal).size(); sample++) {
      h_hcalTimeSample->Fill(sample);
    }
  }

  if (showEmCands) {
    //       std::cout << std::endl << "L1 RCT EmCand objects" << std::endl;
  }
  for (em = rctEmCands->begin(); em != rctEmCands->end(); em++) {
    //  std::cout << "(Analyzer)\n" << (*em) << std::endl;

    L1CaloEmCand *myL1EmCand = new L1CaloEmCand(*em);
    (*myL1EmColl).push_back(*myL1EmCand);
    delete myL1EmCand;

    h_emCandTimeSample->Fill((*em).bx());
    if ((*em).bx() == 0) {
      // std::cout << std::endl << "rank: " << (*em).rank() ;

      if ((*em).rank() > 0) {
        h_emRank->Fill((*em).rank());
        h_emIeta->Fill((*em).regionId().ieta());
        h_emIphi->Fill((*em).regionId().iphi());
        h_emIso->Fill((*em).isolated());
        h_emRankInIetaIphi->Fill((*em).regionId().ieta(), (*em).regionId().iphi(), (*em).rank());
        if ((*em).isolated()) {
          h_emIsoInIetaIphi->Fill((*em).regionId().ieta(), (*em).regionId().iphi());
        } else {
          h_emNonIsoInIetaIphi->Fill((*em).regionId().ieta(), (*em).regionId().iphi());
        }
      }

      if (showEmCands) {
        if ((*em).rank() > 0) {
          //		 std::cout << std::endl << "rank: " << (*em).rank();
          unsigned short rgnPhi = 999;
          unsigned short rgn = (unsigned short)(*em).rctRegion();
          unsigned short card = (unsigned short)(*em).rctCard();
          unsigned short crate = (unsigned short)(*em).rctCrate();

          if (card == 6) {
            rgnPhi = rgn;
          } else if (card < 6) {
            rgnPhi = (card % 2);
          } else {
            std::cout << "rgnPhi not assigned (still " << rgnPhi << ") -- Weird card number! " << card;
          }

          // unsigned short phi_bin = ((crate % 9) * 2) + rgnPhi;
          short eta_bin = (card / 2) * 2 + 1;
          if (card < 6) {
            eta_bin = eta_bin + rgn;
          }
          if (crate < 9) {
            eta_bin = -eta_bin;
          }

          //		   std::cout << /* "rank: " << (*em).rank() << */ "
          // eta_bin: " << eta_bin << "  phi_bin: " << phi_bin << ".  crate: "
          // << crate << "  card: " << card << "  region: " << rgn << ".
          // isolated: " << (*em).isolated();
        }
      }
    } else {
      h_emRankOutOfTime->Fill((*em).rank());
    }
  }
  if (showEmCands) {
    //       std::cout << std::endl;
  }

  // next: SORT THESE GUYS so they're entered into the tree highest first
  //    std::sort(rctEmCands->begin(),rctEmCands->end(),compareEmCands);
  //    for (em=rctEmCands->begin(); em!=rctEmCands->end(); em++)
  std::sort(myL1EmColl->begin(), myL1EmColl->end(), compareEmCands);
  std::reverse(myL1EmColl->begin(), myL1EmColl->end());  // whoops!
  for (em = myL1EmColl->begin(); em != myL1EmColl->end(); em++) {
    emRank.push_back((*em).rank());
    emIeta.push_back((*em).regionId().ieta());
    emIphi.push_back((*em).regionId().iphi());
    emIso.push_back((*em).isolated());
  }
  emTree->Fill();

  if (showRegionSums) {
    std::cout << "Regions" << std::endl;
  }
  for (rgn = rctRegions->begin(); rgn != rctRegions->end(); rgn++) {
    if ((*rgn).bx() == 0) {
      if (showRegionSums && (*rgn).et() > 0) {
        std::cout << /* "(Analyzer)\n" << */ (*rgn) << std::endl;
      }
      if ((*rgn).et() > 0) {
        h_regionSum->Fill((*rgn).et());
        h_regionIeta->Fill((*rgn).gctEta());
        h_regionIphi->Fill((*rgn).gctPhi());
        h_regionSumInIetaIphi->Fill((*rgn).gctEta(), (*rgn).gctPhi(), (*rgn).et());
        h_regionFGInIetaIphi->Fill((*rgn).gctEta(), (*rgn).gctPhi(), (*rgn).fineGrain());
      }
      h_regionMip->Fill((*rgn).mip());
    }
  }
  if (showRegionSums) {
    std::cout << std::endl;
  }

  emRank.clear();
  emIeta.clear();
  emIphi.clear();
  emIso.clear();

  delete myL1EmColl;
}

bool L1RCTTestAnalyzer::compareEmCands(const L1CaloEmCand &cand1, const L1CaloEmCand &cand2) {
  return (cand1.rank() < cand2.rank());
}
