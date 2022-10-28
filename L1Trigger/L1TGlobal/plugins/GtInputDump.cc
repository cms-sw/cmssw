///
/// \class l1t::GtInputDump.cc
///
/// Description: Dump/Analyze Input Collections for GT.
///
/// Implementation:
///    Based off of Michael Mulhearn's YellowParamTester
///
/// \author: Brian Winer Ohio State
///

//
//  This simple module simply retreives the YellowParams object from the event
//  setup, and sends its payload as an INFO message, for debugging purposes.
//

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

using namespace edm;
using namespace std;

namespace l1t {

  // class declaration
  class GtInputDump : public edm::one::EDAnalyzer<> {
  public:
    explicit GtInputDump(const edm::ParameterSet&);
    ~GtInputDump() override{};
    void analyze(const edm::Event&, const edm::EventSetup&) override;

    EDGetToken egToken;
    EDGetToken muToken;
    EDGetToken tauToken;
    EDGetToken jetToken;
    EDGetToken etsumToken;

    int m_minBx;
    int m_maxBx;
  };

  GtInputDump::GtInputDump(const edm::ParameterSet& iConfig) {
    egToken = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("egInputTag"));
    muToken = consumes<BXVector<l1t::Muon>>(iConfig.getParameter<InputTag>("muInputTag"));
    tauToken = consumes<BXVector<l1t::Tau>>(iConfig.getParameter<InputTag>("tauInputTag"));
    jetToken = consumes<BXVector<l1t::Jet>>(iConfig.getParameter<InputTag>("jetInputTag"));
    etsumToken = consumes<BXVector<l1t::EtSum>>(iConfig.getParameter<InputTag>("etsumInputTag"));

    m_minBx = iConfig.getParameter<int>("minBx");
    m_maxBx = iConfig.getParameter<int>("maxBx");
  }

  // loop over events
  void GtInputDump::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
    //inputs
    Handle<BXVector<l1t::EGamma>> egammas;
    iEvent.getByToken(egToken, egammas);

    Handle<BXVector<l1t::Muon>> muons;
    iEvent.getByToken(muToken, muons);

    Handle<BXVector<l1t::Tau>> taus;
    iEvent.getByToken(tauToken, taus);

    Handle<BXVector<l1t::Jet>> jets;
    iEvent.getByToken(jetToken, jets);

    Handle<BXVector<l1t::EtSum>> etsums;
    iEvent.getByToken(etsumToken, etsums);

    printf("\n -------------------------------------- \n");
    printf(" ***********  New Event  ************** \n");
    printf(" -------------------------------------- \n");
    //Loop over BX
    //Loop over BX
    for (int i = m_minBx; i <= m_maxBx; ++i) {
      cout << " ========== BX = " << std::dec << i << " =============================" << endl;

      //Loop over EGamma
      int nObj = 0;
      cout << " ------ EGammas -------- " << endl;
      if (egammas.isValid()) {
        if (i >= egammas->getFirstBX() && i <= egammas->getLastBX()) {
          for (std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(i); eg != egammas->end(i); ++eg) {
            cout << "  " << std::dec << std::setw(2) << std::setfill(' ') << nObj << std::setfill('0') << ")";
            cout << "   Pt " << std::dec << std::setw(3) << eg->hwPt() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << eg->hwPt() << ")";
            cout << "   Eta " << std::dec << std::setw(3) << eg->hwEta() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << (eg->hwEta() & 0xff) << ")";
            cout << "   Phi " << std::dec << std::setw(3) << eg->hwPhi() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << eg->hwPhi() << ")";
            cout << "   Iso " << std::dec << std::setw(1) << eg->hwIso();
            cout << "   Qual " << std::dec << std::setw(1) << eg->hwQual();
            cout << endl;
            nObj++;
          }
        } else {
          cout << "No EG stored for this bx " << i << endl;
        }
      } else {
        cout << "No EG Data in this event " << endl;
      }

      //Loop over Muons
      nObj = 0;
      cout << " ------ Muons --------" << endl;
      if (muons.isValid()) {
        if (i >= muons->getFirstBX() && i <= muons->getLastBX()) {
          for (std::vector<l1t::Muon>::const_iterator mu = muons->begin(i); mu != muons->end(i); ++mu) {
            cout << "  " << std::dec << std::setw(2) << std::setfill(' ') << nObj << std::setfill('0') << ")";
            cout << "   Pt " << std::dec << std::setw(3) << mu->hwPt() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << mu->hwPt() << ")";
            cout << "   EtaAtVtx " << std::dec << std::setw(3) << mu->hwEtaAtVtx() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << (mu->hwEtaAtVtx() & 0x1ff) << ")";
            cout << "   Eta " << std::dec << std::setw(3) << mu->hwEta() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << (mu->hwEta() & 0x1ff) << ")";
            cout << "   PhiAtVtx " << std::dec << std::setw(3) << mu->hwPhiAtVtx() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << mu->hwPhiAtVtx() << ")";
            cout << "   Phi " << std::dec << std::setw(3) << mu->hwPhi() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << mu->hwPhi() << ")";
            cout << "   Iso " << std::dec << std::setw(1) << mu->hwIso();
            cout << "   Qual " << std::dec << std::setw(1) << mu->hwQual();
            cout << endl;
            nObj++;
          }
        } else {
          cout << "No Muons stored for this bx " << i << endl;
        }
      } else {
        cout << "No Muon Data in this event " << endl;
      }

      //Loop over Taus
      nObj = 0;
      cout << " ------ Taus ----------" << endl;
      if (taus.isValid()) {
        if (i >= taus->getFirstBX() && i <= taus->getLastBX()) {
          for (std::vector<l1t::Tau>::const_iterator tau = taus->begin(i); tau != taus->end(i); ++tau) {
            cout << "  " << std::dec << std::setw(2) << std::setfill(' ') << nObj << std::setfill('0') << ")";
            cout << "   Pt " << std::dec << std::setw(3) << tau->hwPt() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << tau->hwPt() << ")";
            cout << "   Eta " << std::dec << std::setw(3) << tau->hwEta() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << (tau->hwEta() & 0xff) << ")";
            cout << "   Phi " << std::dec << std::setw(3) << tau->hwPhi() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << tau->hwPhi() << ")";
            cout << "   Iso " << std::dec << std::setw(1) << tau->hwIso();
            cout << "   Qual " << std::dec << std::setw(1) << tau->hwQual();
            cout << endl;
            nObj++;
          }
        } else {
          cout << "No Taus stored for this bx " << i << endl;
        }
      } else {
        cout << "No Tau Data in this event " << endl;
      }

      //Loop over Jets
      nObj = 0;
      cout << " ------ Jets ----------" << endl;
      if (jets.isValid()) {
        if (i >= jets->getFirstBX() && i <= jets->getLastBX()) {
          for (std::vector<l1t::Jet>::const_iterator jet = jets->begin(i); jet != jets->end(i); ++jet) {
            cout << "  " << std::dec << std::setw(2) << std::setfill(' ') << nObj << std::setfill('0') << ")";
            cout << "   Pt " << std::dec << std::setw(3) << jet->hwPt() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << jet->hwPt() << ")";
            cout << "   Eta " << std::dec << std::setw(3) << jet->hwEta() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << (jet->hwEta() & 0xff) << ")";
            cout << "   Phi " << std::dec << std::setw(3) << jet->hwPhi() << " (0x" << std::hex << std::setw(2)
                 << std::setfill('0') << jet->hwPhi() << ")";
            cout << "   Qual " << std::dec << std::setw(1) << jet->hwQual();
            cout << endl;
            nObj++;
          }
        } else {
          cout << "No Jets stored for this bx " << i << endl;
        }
      } else {
        cout << "No jet Data in this event " << endl;
      }

      //Dump Content
      cout << " ------ EtSums ----------" << endl;
      if (etsums.isValid()) {
        if (i >= etsums->getFirstBX() && i <= etsums->getLastBX()) {
          for (std::vector<l1t::EtSum>::const_iterator etsum = etsums->begin(i); etsum != etsums->end(i); ++etsum) {
            switch (etsum->getType()) {
              case l1t::EtSum::EtSumType::kMissingEt:
                cout << " ETM: ";
                break;
              case l1t::EtSum::EtSumType::kMissingHt:
                cout << " HTM: ";
                break;
              case l1t::EtSum::EtSumType::kTotalEt:
                cout << " ETT: ";
                break;
              case l1t::EtSum::EtSumType::kTotalHt:
                cout << " HTT: ";
                break;
              default:
                cout << " Unknown: ";
                break;
            }
            cout << " Et " << std::dec << std::setw(3) << etsum->hwPt() << " (0x" << std::hex << std::setw(3)
                 << std::setfill('0') << etsum->hwPt() << ")";
            if (etsum->getType() == l1t::EtSum::EtSumType::kMissingEt ||
                etsum->getType() == l1t::EtSum::EtSumType::kMissingHt)
              cout << " Phi " << std::dec << std::setw(3) << etsum->hwPhi() << " (0x" << std::hex << std::setw(2)
                   << std::setfill('0') << etsum->hwPhi() << ")";
            cout << endl;
          }
        } else {
          cout << "No EtSums stored for this bx " << i << endl;
        }
      } else {
        cout << "No EtSum Data in this event " << endl;
      }
    }
    printf("\n");
  }

}  // namespace l1t

DEFINE_FWK_MODULE(l1t::GtInputDump);
