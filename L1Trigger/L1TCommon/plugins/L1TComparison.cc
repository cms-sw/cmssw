// -*- C++ -*-
//
// L1TComparison:  produce a summary of comparison of L1T event data
//

#include <iostream>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

using namespace std;
using namespace edm;
using namespace l1t;

static bool compare_l1candidate(const L1Candidate& a, const L1Candidate& b, int verbose = 1) {
  int status = 0;
  if (a.pt() != b.pt())
    status = 1;
  if (a.eta() != b.eta())
    status = 1;
  if (a.phi() != b.phi())
    status = 1;

  if (status) {
    cout << "COMPARISON FAILURE:  \n";
    cout << "A:  pt = " << a.pt() << " eta = " << a.eta() << " phi = " << a.phi() << "\n";
    cout << "B:  pt = " << b.pt() << " eta = " << b.eta() << " phi = " << b.phi() << "\n";
  }

  if (a.hwPt() != b.hwPt())
    status = 1;
  if (a.hwEta() != b.hwEta())
    status = 1;
  if (a.hwPhi() != b.hwPhi())
    status = 1;

  if (status) {
    cout << "COMPARISON FAILURE:  \n";
    cout << "A:  hwPt = " << a.hwPt() << " hwEta = " << a.hwEta() << " hwPhi = " << a.hwPhi() << "\n";
    cout << "B:  hwPt = " << b.hwPt() << " hwEta = " << b.hwEta() << " hwPhi = " << b.hwPhi() << "\n";
  }

  if (a.hwQual() != b.hwQual())
    status = 1;
  if (a.hwIso() != b.hwIso())
    status = 1;
  if (status) {
    cout << "COMPARISON FAILURE:  \n";
    cout << "A:  hwQual = " << a.hwQual() << " hwIso = " << a.hwIso() << "\n";
    cout << "B:  hwQual = " << b.hwQual() << " hwIso = " << b.hwIso() << "\n";
  }

  return status;
}

class L1TComparison : public one::EDAnalyzer<> {
public:
  explicit L1TComparison(const ParameterSet&);
  ~L1TComparison() override;

  static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(Event const&, EventSetup const&) override;
  void endJob() override;

  // Tag string to mark summary with:
  string tag_;

  // Checks to perform:
  bool egCheck_;
  bool tauCheck_;
  bool jetCheck_;
  bool sumCheck_;
  bool muonCheck_;
  bool algCheck_;
  bool bxZeroOnly_;

  // EDM tokens:
  edm::EDGetTokenT<EGammaBxCollection> egTokenA_;
  edm::EDGetTokenT<TauBxCollection> tauTokenA_;
  edm::EDGetTokenT<JetBxCollection> jetTokenA_;
  edm::EDGetTokenT<EtSumBxCollection> sumTokenA_;
  edm::EDGetTokenT<MuonBxCollection> muonTokenA_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> algTokenA_;

  edm::EDGetTokenT<EGammaBxCollection> egTokenB_;
  edm::EDGetTokenT<TauBxCollection> tauTokenB_;
  edm::EDGetTokenT<JetBxCollection> jetTokenB_;
  edm::EDGetTokenT<EtSumBxCollection> sumTokenB_;
  edm::EDGetTokenT<MuonBxCollection> muonTokenB_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> algTokenB_;

  // keep a tally for summary:
  int egCount_;
  int tauCount_;
  int jetCount_;
  int sumCount_;
  int muonCount_;
  int algCount_;

  int egFails_;
  int tauFails_;
  int jetFails_;
  int sumFails_;
  int muonFails_;
  int algFails_;
};

L1TComparison::L1TComparison(const ParameterSet& iConfig) {
  tag_ = iConfig.getParameter<string>("tag");
  egCheck_ = iConfig.getParameter<bool>("egCheck");
  tauCheck_ = iConfig.getParameter<bool>("tauCheck");
  jetCheck_ = iConfig.getParameter<bool>("jetCheck");
  sumCheck_ = iConfig.getParameter<bool>("sumCheck");
  muonCheck_ = iConfig.getParameter<bool>("muonCheck");
  algCheck_ = iConfig.getParameter<bool>("algCheck");
  bxZeroOnly_ = iConfig.getParameter<bool>("bxZeroOnly");

  cout << "L1T Summary for " << tag_ << "\n";
  cout << "DEBUG:  egCheck:    " << egCheck_ << "\n";
  cout << "DEBUG:  tauCheck:   " << tauCheck_ << "\n";
  cout << "DEBUG:  jetCheck:   " << jetCheck_ << "\n";
  cout << "DEBUG:  sumCheck:   " << sumCheck_ << "\n";
  cout << "DEBUG:  muonCheck:  " << muonCheck_ << "\n";
  cout << "DEBUG:  algCheck:   " << algCheck_ << "\n";

  if (egCheck_) {
    egTokenA_ = consumes<EGammaBxCollection>(iConfig.getParameter<InputTag>("egTagA"));
  }
  if (tauCheck_) {
    tauTokenA_ = consumes<TauBxCollection>(iConfig.getParameter<InputTag>("tauTagA"));
  }
  if (jetCheck_) {
    jetTokenA_ = consumes<JetBxCollection>(iConfig.getParameter<InputTag>("jetTagA"));
  }
  if (sumCheck_) {
    sumTokenA_ = consumes<EtSumBxCollection>(iConfig.getParameter<InputTag>("sumTagA"));
  }
  if (muonCheck_) {
    muonTokenA_ = consumes<MuonBxCollection>(iConfig.getParameter<InputTag>("muonTagA"));
  }
  if (algCheck_) {
    algTokenA_ = consumes<GlobalAlgBlkBxCollection>(iConfig.getParameter<InputTag>("algTagA"));
  }

  if (egCheck_) {
    egTokenB_ = consumes<EGammaBxCollection>(iConfig.getParameter<InputTag>("egTagB"));
  }
  if (tauCheck_) {
    tauTokenB_ = consumes<TauBxCollection>(iConfig.getParameter<InputTag>("tauTagB"));
  }
  if (jetCheck_) {
    jetTokenB_ = consumes<JetBxCollection>(iConfig.getParameter<InputTag>("jetTagB"));
  }
  if (sumCheck_) {
    sumTokenB_ = consumes<EtSumBxCollection>(iConfig.getParameter<InputTag>("sumTagB"));
  }
  if (muonCheck_) {
    muonTokenB_ = consumes<MuonBxCollection>(iConfig.getParameter<InputTag>("muonTagB"));
  }
  if (algCheck_) {
    algTokenB_ = consumes<GlobalAlgBlkBxCollection>(iConfig.getParameter<InputTag>("algTagB"));
  }

  egCount_ = 0;
  tauCount_ = 0;
  jetCount_ = 0;
  sumCount_ = 0;
  muonCount_ = 0;
  algCount_ = 0;

  egFails_ = 0;
  tauFails_ = 0;
  jetFails_ = 0;
  sumFails_ = 0;
  muonFails_ = 0;
  algFails_ = 0;
}

L1TComparison::~L1TComparison() {}

void L1TComparison::analyze(Event const& iEvent, EventSetup const& iSetup) {
  cout << "L1TComparison Module output for " << tag_ << "\n";

  if (egCheck_) {
    Handle<EGammaBxCollection> XTMPA;
    iEvent.getByToken(egTokenA_, XTMPA);
    Handle<EGammaBxCollection> XTMPB;
    iEvent.getByToken(egTokenB_, XTMPB);

    if (!(XTMPA.isValid() && XTMPB.isValid())) {
      LogWarning("MissingProduct") << "L1Upgrade e-gamma's not found." << std::endl;
    } else {
      for (int ibx = XTMPA->getFirstBX(); ibx <= XTMPA->getLastBX(); ++ibx) {
        if (bxZeroOnly_ && (ibx != 0))
          continue;
        if (ibx < XTMPB->getFirstBX())
          continue;
        if (ibx > XTMPB->getLastBX())
          continue;
        int sizeA = XTMPA->size(ibx);
        int sizeB = XTMPB->size(ibx);
        if (sizeA != sizeB) {
          cout << "L1T COMPARISON FAILURE:  collections have different sizes for bx = " << ibx << "\n";
        } else {
          auto itB = XTMPB->begin(ibx);
          for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
            bool fail = compare_l1candidate(*itA, *itB);
            itB++;
            if (!fail) {
              egCount_++;
            } else {
              egFails_++;
            }
          }
        }
      }
    }
  }

  if (tauCheck_) {
    Handle<TauBxCollection> XTMPA;
    iEvent.getByToken(tauTokenA_, XTMPA);
    Handle<TauBxCollection> XTMPB;
    iEvent.getByToken(tauTokenB_, XTMPB);

    if (!(XTMPA.isValid() && XTMPB.isValid())) {
      LogWarning("MissingProduct") << "L1Upgrade tau's not found." << std::endl;
    } else {
      for (int ibx = XTMPA->getFirstBX(); ibx <= XTMPA->getLastBX(); ++ibx) {
        if (bxZeroOnly_ && (ibx != 0))
          continue;
        if (ibx < XTMPB->getFirstBX())
          continue;
        if (ibx > XTMPB->getLastBX())
          continue;
        int sizeA = XTMPA->size(ibx);
        int sizeB = XTMPB->size(ibx);
        if (sizeA != sizeB) {
          cout << "L1T COMPARISON FAILURE:  collections have different sizes for bx = " << ibx << "\n";
        } else {
          auto itB = XTMPB->begin(ibx);
          for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
            bool fail = compare_l1candidate(*itA, *itB);
            itB++;
            if (!fail) {
              tauCount_++;
            } else {
              tauFails_++;
            }
          }
        }
      }
    }
  }

  if (jetCheck_) {
    Handle<JetBxCollection> XTMPA;
    iEvent.getByToken(jetTokenA_, XTMPA);
    Handle<JetBxCollection> XTMPB;
    iEvent.getByToken(jetTokenB_, XTMPB);

    if (!(XTMPA.isValid() && XTMPB.isValid())) {
      LogWarning("MissingProduct") << "L1Upgrade jet's not found." << std::endl;
    } else {
      for (int ibx = XTMPA->getFirstBX(); ibx <= XTMPA->getLastBX(); ++ibx) {
        if (bxZeroOnly_ && (ibx != 0))
          continue;
        if (ibx < XTMPB->getFirstBX())
          continue;
        if (ibx > XTMPB->getLastBX())
          continue;
        int sizeA = XTMPA->size(ibx);
        int sizeB = XTMPB->size(ibx);
        if (sizeA != sizeB) {
          cout << "L1T COMPARISON FAILURE:  collections have different sizes for bx = " << ibx << "\n";
        } else {
          auto itB = XTMPB->begin(ibx);
          for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
            bool fail = compare_l1candidate(*itA, *itB);
            itB++;
            if (!fail) {
              jetCount_++;
            } else {
              jetFails_++;
            }
          }
        }
      }
    }
  }

  if (sumCheck_) {
    Handle<EtSumBxCollection> XTMPA;
    iEvent.getByToken(sumTokenA_, XTMPA);
    Handle<EtSumBxCollection> XTMPB;
    iEvent.getByToken(sumTokenB_, XTMPB);

    if (!(XTMPA.isValid() && XTMPB.isValid())) {
      LogWarning("MissingProduct") << "L1Upgrade sum's not found." << std::endl;
    } else {
      for (int ibx = XTMPA->getFirstBX(); ibx <= XTMPA->getLastBX(); ++ibx) {
        if (bxZeroOnly_ && (ibx != 0))
          continue;
        if (ibx < XTMPB->getFirstBX())
          continue;
        if (ibx > XTMPB->getLastBX())
          continue;
        int sizeA = XTMPA->size(ibx);
        int sizeB = XTMPB->size(ibx);

        if (sizeA != sizeB) {
          cout << "L1T COMPARISON WARNING:  sums collections have different sizes for bx = " << ibx << "\n";
          cout << "L1T COMPARISON WARNING:  sums collections A size  = " << sizeA
               << "  sums collection B size = " << sizeB << "\n";
          cout << "L1T COMPARISON WARNING:  known issue because packer has not been udpated for Minbias\n";
        }
        for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
          cout << "L1T COMPARISON :  EtSum type: A = " << itA->getType() << "\n";
        }
        for (auto itB = XTMPB->begin(ibx); itB != XTMPB->end(ibx); ++itB) {
          cout << "L1T COMPARISON :  EtSum type: B = " << itB->getType() << "\n";
        }

        // temp workaround for sums not packed...
        if (sizeA > sizeB)
          sizeA = sizeB;
        if (sizeB > sizeA)
          sizeB = sizeA;

        if (sizeA != sizeB) {
          cout << "L1T COMPARISON FAILURE:  collections have different sizes for bx = " << ibx << "\n";
        } else {
          auto itB = XTMPB->begin(ibx);
          for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
            cout << "L1T COMPARISON :  EtSum type: A = " << itA->getType() << " vs B = " << itB->getType() << "\n";
            if (itA->getType() != itB->getType()) {
              cout << "L1T COMPARISON FAILURE:  Different types .... EtSum type:" << itA->getType() << " vs "
                   << itB->getType() << "\n";
            }
            if (itA->getType() == EtSum::kTotalEtEm)
              cout << "L1T COMPARISON WARNING:  (known issue) sum of type " << itA->getType()
                   << " when emulated has a dummy value (pending proper emulation)"
                   << "\n";
            if (itA->getType() < EtSum::kMinBiasHFP0 || itA->getType() > EtSum::kMinBiasHFM1) {
              bool fail = compare_l1candidate(*itA, *itB);
              if (fail) {
                cout << "L1T COMPARISON FAILURE:  for type " << itA->getType() << "\n";
              }
              if (!fail) {
                sumCount_++;
              } else {
                sumFails_++;
              }
            } else {
              cout << "L1T COMPARISON WARNING:  (known issue) not checking sum of type " << itA->getType() << "\n";
            }
            itB++;
          }
        }
      }
    }
  }

  if (muonCheck_) {
    Handle<MuonBxCollection> XTMPA;
    iEvent.getByToken(muonTokenA_, XTMPA);
    Handle<MuonBxCollection> XTMPB;
    iEvent.getByToken(muonTokenB_, XTMPB);

    if (!(XTMPA.isValid() && XTMPB.isValid())) {
      LogWarning("MissingProduct") << "L1Upgrade muon's not found." << std::endl;
    } else {
      for (int ibx = XTMPA->getFirstBX(); ibx <= XTMPA->getLastBX(); ++ibx) {
        if (bxZeroOnly_ && (ibx != 0))
          continue;
        if (ibx < XTMPB->getFirstBX())
          continue;
        if (ibx > XTMPB->getLastBX())
          continue;
        int sizeA = XTMPA->size(ibx);
        int sizeB = XTMPB->size(ibx);
        if (sizeA != sizeB) {
          cout << "L1T COMPARISON FAILURE:  collections have different sizes for bx = " << ibx << "\n";
        } else {
          auto itB = XTMPB->begin(ibx);
          for (auto itA = XTMPA->begin(ibx); itA != XTMPA->end(ibx); ++itA) {
            bool fail = compare_l1candidate(*itA, *itB);
            itB++;
            if (!fail) {
              muonCount_++;
            } else {
              muonFails_++;
            }
          }
        }
      }
    }
  }
}

void L1TComparison::beginJob() { cout << "INFO:  L1TComparison module beginJob called.\n"; }

void L1TComparison::endJob() {
  cout << "INFO:  L1T Comparison for " << tag_ << "\n";
  cout << "INFO: count of successful comparison for each type follows:\n";
  if (egCheck_)
    cout << "eg:    " << egCount_ << "\n";
  if (tauCheck_)
    cout << "tau:   " << tauCount_ << "\n";
  if (jetCheck_)
    cout << "jet:   " << jetCount_ << "\n";
  if (sumCheck_)
    cout << "sum:   " << sumCount_ << "\n";
  if (muonCheck_)
    cout << "muon:  " << muonCount_ << "\n";
  cout << "INFO: count of failed comparison for each type follows:\n";
  if (egCheck_)
    cout << "eg:    " << egFails_ << "\n";
  if (tauCheck_)
    cout << "tau:   " << tauFails_ << "\n";
  if (jetCheck_)
    cout << "jet:   " << jetFails_ << "\n";
  if (sumCheck_)
    cout << "sum:   " << sumFails_ << "\n";
  if (muonCheck_)
    cout << "muon:  " << muonFails_ << "\n";

  int fail = 0;
  if (egCheck_ && ((egFails_ > 0) || (egCount_ <= 0)))
    fail = 1;
  if (tauCheck_ && ((tauFails_ > 0) || (tauCount_ <= 0)))
    fail = 1;
  if (jetCheck_ && ((jetFails_ > 0) || (jetCount_ <= 0)))
    fail = 1;
  if (sumCheck_ && ((sumFails_ > 0) || (sumCount_ <= 0)))
    fail = 1;
  if (muonCheck_ && ((muonFails_ > 0) || (muonCount_ <= 0)))
    fail = 1;

  if (fail) {
    cout << "SUMMARY:  L1T Comparison for " << tag_ << " was FAILURE\n";
  } else {
    cout << "SUMMARY:  L1T Comparison for " << tag_ << " was SUCCESS\n";
  }
}

void L1TComparison::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1TComparison);
