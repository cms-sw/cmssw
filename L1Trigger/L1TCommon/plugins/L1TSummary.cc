// -*- C++ -*-
//
// L1TSummary:  produce command line visible summary of L1T system
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

using namespace std;
using namespace edm;
using namespace l1t;

class L1TSummary : public one::EDAnalyzer<> {
public:
  explicit L1TSummary(const ParameterSet&);

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
  bool bxZeroOnly_;

  // EDM tokens:
  edm::EDGetTokenT<EGammaBxCollection> egToken_;
  std::vector<edm::EDGetTokenT<TauBxCollection>> tauTokens_;
  edm::EDGetTokenT<JetBxCollection> jetToken_;
  edm::EDGetTokenT<EtSumBxCollection> sumToken_;
  edm::EDGetTokenT<MuonBxCollection> muonToken_;

  // keep a tally for summary:
  int egCount_;
  int tauCount_;
  int jetCount_;
  int sumCount_;
  int muonCount_;
};

L1TSummary::L1TSummary(const ParameterSet& iConfig) {
  // InputTag barrelTfInputTag = iConfig.getParameter<InputTag>("barrelTFInput");
  // InputTag overlapTfInputTag = iConfig.getParameter<InputTag>("overlapTFInput");
  // InputTag forwardTfInputTag = iConfig.getParameter<InputTag>("forwardTFInput");
  //m_barrelTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(iConfig.getParameter<InputTag>("bmtfDigis"));

  tag_ = iConfig.getParameter<string>("tag");

  egCheck_ = iConfig.getParameter<bool>("egCheck");
  tauCheck_ = iConfig.getParameter<bool>("tauCheck");
  jetCheck_ = iConfig.getParameter<bool>("jetCheck");
  sumCheck_ = iConfig.getParameter<bool>("sumCheck");
  muonCheck_ = iConfig.getParameter<bool>("muonCheck");
  bxZeroOnly_ = iConfig.getParameter<bool>("bxZeroOnly");

  //cout << "L1T Summary for " << tag << "\n";
  //cout << "DEBUG:  egCheck:    " << egCheck_ << "\n";
  //cout << "DEBUG:  tauCheck:   " << tauCheck_ << "\n";
  //cout << "DEBUG:  jetCheck:   " << jetCheck_ << "\n";
  //cout << "DEBUG:  sumCheck:   " << sumCheck_ << "\n";
  //cout << "DEBUG:  muonCheck:  " << muonCheck_ << "\n";

  if (egCheck_) {
    egToken_ = consumes<EGammaBxCollection>(iConfig.getParameter<InputTag>("egToken"));
  }
  if (tauCheck_) {
    const auto& taus = iConfig.getParameter<std::vector<edm::InputTag>>("tauTokens");
    for (const auto& tau : taus) {
      tauTokens_.push_back(consumes<l1t::TauBxCollection>(tau));
    }
  }
  if (jetCheck_) {
    jetToken_ = consumes<JetBxCollection>(iConfig.getParameter<InputTag>("jetToken"));
  }
  if (sumCheck_) {
    sumToken_ = consumes<EtSumBxCollection>(iConfig.getParameter<InputTag>("sumToken"));
  }
  if (muonCheck_) {
    muonToken_ = consumes<MuonBxCollection>(iConfig.getParameter<InputTag>("muonToken"));
  }

  egCount_ = 0;
  tauCount_ = 0;
  jetCount_ = 0;
  sumCount_ = 0;
  muonCount_ = 0;
}

void L1TSummary::analyze(Event const& iEvent, EventSetup const& iSetup) {
  cout << "L1TSummary Module output for " << tag_ << "\n";
  if (egCheck_) {
    Handle<EGammaBxCollection> XTMP;
    iEvent.getByToken(egToken_, XTMP);
    if (XTMP.isValid()) {
      cout << "INFO:  L1T found e-gamma collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
        for (auto it = XTMP->begin(ibx); it != XTMP->end(ibx); it++) {
          if (bxZeroOnly_ && (ibx != 0))
            continue;
          if (it->et() > 0) {
            egCount_++;
            cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi()
                 << "\n";
          }
        }
      }
    } else {
      LogWarning("MissingProduct") << "L1Upgrade e-gamma's not found." << std::endl;
    }
  }

  if (tauCheck_) {
    for (auto& tautoken : tauTokens_) {
      Handle<TauBxCollection> XTMP;
      iEvent.getByToken(tautoken, XTMP);
      if (XTMP.isValid()) {
        cout << "INFO:  L1T found tau collection.\n";
        for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
          for (auto it = XTMP->begin(ibx); it != XTMP->end(ibx); it++) {
            if (it->et() > 0) {
              if (bxZeroOnly_ && (ibx != 0))
                continue;
              tauCount_++;
              cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi()
                   << "\n";
            }
          }
        }
      } else {
        LogWarning("MissingProduct") << "L1Upgrade tau's not found." << std::endl;
      }
    }
  }

  if (jetCheck_) {
    Handle<JetBxCollection> XTMP;
    iEvent.getByToken(jetToken_, XTMP);
    if (XTMP.isValid()) {
      cout << "INFO:  L1T found jet collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
        for (auto it = XTMP->begin(ibx); it != XTMP->end(ibx); it++) {
          if (it->et() > 0) {
            if (bxZeroOnly_ && (ibx != 0))
              continue;
            jetCount_++;
            cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi()
                 << "\n";
          }
        }
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade jets not found." << std::endl;
    }
  }

  if (sumCheck_) {
    Handle<EtSumBxCollection> XTMP;
    iEvent.getByToken(sumToken_, XTMP);
    if (XTMP.isValid()) {
      cout << "INFO:  L1T found sum collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
        for (auto it = XTMP->begin(ibx); it != XTMP->end(ibx); it++) {
          //if (it->et() > 0) {
          if (bxZeroOnly_ && (ibx != 0))
            continue;
          sumCount_++;
          cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi()
               << " type:  " << it->getType() << "\n";
          //}
        }
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade sums not found." << std::endl;
    }
  }

  if (muonCheck_) {
    Handle<MuonBxCollection> XTMP;
    iEvent.getByToken(muonToken_, XTMP);
    if (XTMP.isValid()) {
      cout << "INFO:  L1T found muon collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
        for (auto it = XTMP->begin(ibx); it != XTMP->end(ibx); it++) {
          if (it->et() > 0) {
            if (bxZeroOnly_ && (ibx != 0))
              continue;
            muonCount_++;
            cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi()
                 << "\n";
          }
        }
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade muons not found." << std::endl;
    }
  }
}

void L1TSummary::beginJob() { cout << "INFO:  L1TSummary module beginJob called.\n"; }

void L1TSummary::endJob() {
  cout << "INFO:  L1T Summary for " << tag_ << "\n";
  cout << "INFO: count of non-zero candidates for each type follows:\n";
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
}

void L1TSummary::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1TSummary);
