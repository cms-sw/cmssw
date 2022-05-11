// -*- C++ -*-
//
// L1TBasicDemo:  demonstrate basic access of L1T objects
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/EtSumHelper.h"

// leaving out the following namespaces, so namespaces are explicit in the demo code:
// using namespace l1t;
// using namespace edm;

class L1TBasicDemo : public edm::global::EDAnalyzer<> {
public:
  explicit L1TBasicDemo(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  // EDM tokens:
  edm::EDGetTokenT<l1t::EGammaBxCollection> egToken_;
  edm::EDGetTokenT<l1t::TauBxCollection> tauToken_;
  edm::EDGetTokenT<l1t::JetBxCollection> jetToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> sumToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> muonToken_;

  int trigger_bx_only;
};

L1TBasicDemo::L1TBasicDemo(const edm::ParameterSet& iConfig) {
  egToken_ = consumes<l1t::EGammaBxCollection>(iConfig.getParameter<edm::InputTag>("EgTag"));
  tauToken_ = consumes<l1t::TauBxCollection>(iConfig.getParameter<edm::InputTag>("TauTag"));
  jetToken_ = consumes<l1t::JetBxCollection>(iConfig.getParameter<edm::InputTag>("JetTag"));
  sumToken_ = consumes<l1t::EtSumBxCollection>(iConfig.getParameter<edm::InputTag>("SumTag"));
  muonToken_ = consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("MuonTag"));
  trigger_bx_only = iConfig.getParameter<bool>("UseTriggerBxOnly");
}

void L1TBasicDemo::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  cout << "INFO:  dumping EGamma BX collection:\n";
  edm::Handle<l1t::EGammaBxCollection> eg;
  iEvent.getByToken(egToken_, eg);
  if (eg.isValid()) {
    for (int ibx = eg->getFirstBX(); ibx <= eg->getLastBX(); ++ibx) {
      if (trigger_bx_only && (ibx != 0))
        continue;
      for (auto it = eg->begin(ibx); it != eg->end(ibx); it++) {
        if (it->et() == 0)
          continue;  // if you don't care about L1T candidates with zero ET.
        cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi() << "\n";
      }
    }
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade e-gamma bx collection not found." << std::endl;
  }

  cout << "INFO:  dumping Tau BX collection:\n";
  edm::Handle<l1t::TauBxCollection> tau;
  iEvent.getByToken(tauToken_, tau);
  if (tau.isValid()) {
    for (int ibx = tau->getFirstBX(); ibx <= tau->getLastBX(); ++ibx) {
      if (trigger_bx_only && (ibx != 0))
        continue;
      for (auto it = tau->begin(ibx); it != tau->end(ibx); it++) {
        if (it->et() == 0)
          continue;  // if you don't care about L1T candidates with zero ET.
        cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi() << "\n";
      }
    }
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade tau bx collection not found." << std::endl;
  }

  cout << "INFO:  dumping Jet BX collection:\n";
  edm::Handle<l1t::JetBxCollection> jet;
  iEvent.getByToken(jetToken_, jet);
  if (jet.isValid()) {
    for (int ibx = jet->getFirstBX(); ibx <= jet->getLastBX(); ++ibx) {
      if (trigger_bx_only && (ibx != 0))
        continue;
      for (auto it = jet->begin(ibx); it != jet->end(ibx); it++) {
        if (it->et() == 0)
          continue;  // if you don't care about L1T candidates with zero ET.
        cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi() << "\n";
      }
    }
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade jet bx collection not found." << std::endl;
  }

  cout << "INFO:  dumping EtSum BX collection:\n";
  edm::Handle<l1t::EtSumBxCollection> sum;
  iEvent.getByToken(sumToken_, sum);
  if (sum.isValid()) {
    l1t::EtSumHelper hsum(sum);
    cout << "met:     " << hsum.MissingEt() << "\n";
    cout << "met phi: " << hsum.MissingEtPhi() << "\n";
    cout << "mht:     " << hsum.MissingHt() << "\n";
    cout << "mht phi: " << hsum.MissingHtPhi() << "\n";
    cout << "sum et:  " << hsum.TotalEt() << "\n";
    cout << "sum ht:  " << hsum.TotalHt() << "\n";
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade sum bx collection not found." << std::endl;
  }

  cout << "INFO:  dumping Muon BX collection:\n";
  edm::Handle<l1t::MuonBxCollection> muon;
  iEvent.getByToken(muonToken_, muon);
  if (muon.isValid()) {
    for (int ibx = muon->getFirstBX(); ibx <= muon->getLastBX(); ++ibx) {
      if (trigger_bx_only && (ibx != 0))
        continue;
      for (auto it = muon->begin(ibx); it != muon->end(ibx); it++) {
        if (it->et() == 0)
          continue;  // if you don't care about L1T candidates with zero ET.
        cout << "bx:  " << ibx << "  et:  " << it->et() << "  eta:  " << it->eta() << "  phi:  " << it->phi() << "\n";
      }
    }
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade muon bx collection not found." << std::endl;
  }
}

void L1TBasicDemo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1TBasicDemo);
