///
/// \class l1t::DumpMuonScouting.cc
///
/// Description: Dump/Analyze Moun Scouting stored in BXVector
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

// system include files
#include <fstream>
#include <iomanip>
#include <memory>

// user include files
//   base class
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

using namespace edm;
using namespace std;

// class declaration
class DumpMuonScouting : public edm::stream::EDAnalyzer<> {
public:
  explicit DumpMuonScouting(const edm::ParameterSet&);
  ~DumpMuonScouting() override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  EDGetTokenT<BXVector<l1t::Muon>> muToken;

  int m_minBx;
  int m_maxBx;

private:
  int m_tvVersion;
};

DumpMuonScouting::DumpMuonScouting(const edm::ParameterSet& iConfig) {
  muToken = consumes<l1t::MuonBxCollection>(iConfig.getParameter<InputTag>("muInputTag"));

  m_minBx = iConfig.getParameter<int>("minBx");
  m_maxBx = iConfig.getParameter<int>("maxBx");
}

// loop over events
void DumpMuonScouting::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  //input
  Handle<BXVector<l1t::Muon>> muons = iEvent.getHandle(muToken);

  {
    cout << " -----------------------------------------------------  " << endl;
    cout << " *********** Run " << std::dec << iEvent.id().run() << " Event " << iEvent.id().event()
         << " **************  " << endl;
    cout << " ----------------------------------------------------- " << endl;

    //Loop over BX
    for (int i = m_minBx; i <= m_maxBx; ++i) {
      //Loop over Muons
      //cout << " ------ Muons --------" << endl;
      if (muons.isValid()) {
        if (i >= muons->getFirstBX() && i <= muons->getLastBX()) {
          for (std::vector<l1t::Muon>::const_iterator mu = muons->begin(i); mu != muons->end(i); ++mu) {
            cout << "  " << std::dec << std::setw(2) << std::setfill(' ') << std::setfill('0') << ")";
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
            cout << "   Chrg " << std::dec << std::setw(1) << mu->hwCharge();
            cout << endl;
          }
        } else {
          cout << "No Muons stored for this bx " << i << endl;
        }
      } else {
        cout << "No Muon Data in this event " << endl;
      }

    }  //loop over Bx
    cout << std::dec << endl;
  }  //if dumpGtRecord
}

DEFINE_FWK_MODULE(DumpMuonScouting);
