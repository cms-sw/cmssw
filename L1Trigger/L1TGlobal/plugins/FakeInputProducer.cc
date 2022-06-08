///
/// \class l1t::FakeInputProducer
///
/// Description: Create Fake Input Collections for the GT.  Allows testing of emulation
///
///
/// \author: B. Winer OSU
///
///  Modeled after FakeInputProducer

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

using namespace std;
using namespace edm;

namespace l1t {

  //
  // class declaration
  //

  class FakeInputProducer : public global::EDProducer<> {
  public:
    explicit FakeInputProducer(const ParameterSet&);
    ~FakeInputProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(StreamID, Event&, EventSetup const&) const override;

    // ----------member data ---------------------------
    //std::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //std::shared_ptr<const FirmwareVersion> m_fwv;
    //std::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    // Parameters for EG
    std::vector<int> fEgBx;
    std::vector<int> fEgHwPt;
    std::vector<int> fEgHwPhi;
    std::vector<int> fEgHwEta;
    std::vector<int> fEgIso;

    // Parameters for Mu
    std::vector<int> fMuBx;
    std::vector<int> fMuHwPt;
    std::vector<int> fMuHwPhi;
    std::vector<int> fMuHwEta;
    std::vector<int> fMuIso;

    // Parameters for Tau
    std::vector<int> fTauBx;
    std::vector<int> fTauHwPt;
    std::vector<int> fTauHwPhi;
    std::vector<int> fTauHwEta;
    std::vector<int> fTauIso;

    // Parameters for Jet
    std::vector<int> fJetBx;
    std::vector<int> fJetHwPt;
    std::vector<int> fJetHwPhi;
    std::vector<int> fJetHwEta;

    // Parameters for EtSum
    std::vector<int> fEtSumBx;
    std::vector<int> fEtSumHwPt;
    std::vector<int> fEtSumHwPhi;
  };

  //
  // constructors and destructor
  //
  FakeInputProducer::FakeInputProducer(const ParameterSet& iConfig) {
    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Muon>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();

    // Setup Parameter Set for EG
    ParameterSet eg_params = iConfig.getUntrackedParameter<ParameterSet>("egParams");

    fEgBx = eg_params.getUntrackedParameter<vector<int>>("egBx");
    fEgHwPt = eg_params.getUntrackedParameter<vector<int>>("egHwPt");
    fEgHwPhi = eg_params.getUntrackedParameter<vector<int>>("egHwPhi");
    fEgHwEta = eg_params.getUntrackedParameter<vector<int>>("egHwEta");
    fEgIso = eg_params.getUntrackedParameter<vector<int>>("egIso");

    // Setup Parameter Set for Muon
    ParameterSet mu_params = iConfig.getUntrackedParameter<ParameterSet>("muParams");

    fMuBx = mu_params.getUntrackedParameter<vector<int>>("muBx");
    fMuHwPt = mu_params.getUntrackedParameter<vector<int>>("muHwPt");
    fMuHwPhi = mu_params.getUntrackedParameter<vector<int>>("muHwPhi");
    fMuHwEta = mu_params.getUntrackedParameter<vector<int>>("muHwEta");
    fMuIso = mu_params.getUntrackedParameter<vector<int>>("muIso");

    // Setup Parameter Set for taus
    ParameterSet tau_params = iConfig.getUntrackedParameter<ParameterSet>("tauParams");

    fTauBx = tau_params.getUntrackedParameter<vector<int>>("tauBx");
    fTauHwPt = tau_params.getUntrackedParameter<vector<int>>("tauHwPt");
    fTauHwPhi = tau_params.getUntrackedParameter<vector<int>>("tauHwPhi");
    fTauHwEta = tau_params.getUntrackedParameter<vector<int>>("tauHwEta");
    fTauIso = tau_params.getUntrackedParameter<vector<int>>("tauIso");

    // Setup Parameter Set for jet
    ParameterSet jet_params = iConfig.getUntrackedParameter<ParameterSet>("jetParams");

    fJetBx = jet_params.getUntrackedParameter<vector<int>>("jetBx");
    fJetHwPt = jet_params.getUntrackedParameter<vector<int>>("jetHwPt");
    fJetHwPhi = jet_params.getUntrackedParameter<vector<int>>("jetHwPhi");
    fJetHwEta = jet_params.getUntrackedParameter<vector<int>>("jetHwEta");

    // Setup Parameter Set for EtSums
    ParameterSet etsum_params = iConfig.getUntrackedParameter<ParameterSet>("etsumParams");

    fEtSumBx = etsum_params.getUntrackedParameter<vector<int>>("etsumBx");
    fEtSumHwPt = etsum_params.getUntrackedParameter<vector<int>>("etsumHwPt");
    fEtSumHwPhi = etsum_params.getUntrackedParameter<vector<int>>("etsumHwPhi");
  }

  FakeInputProducer::~FakeInputProducer() {}

  //
  // member functions
  //

  // ------------ method called to produce the data ------------
  void FakeInputProducer::produce(StreamID, Event& iEvent, const EventSetup& iSetup) const {
    LogDebug("l1t|Global") << "FakeInputProducer::produce function called...\n";

    // Set the range of BX....TO DO...move to Params or determine from param set.
    int bxFirst = -2;
    int bxLast = 2;

    //outputs
    std::unique_ptr<l1t::EGammaBxCollection> egammas(new l1t::EGammaBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::MuonBxCollection> muons(new l1t::MuonBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::TauBxCollection> taus(new l1t::TauBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::JetBxCollection> jets(new l1t::JetBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::EtSumBxCollection> etsums(new l1t::EtSumBxCollection(0, bxFirst, bxLast));

    // Put EG into Collections
    for (unsigned int it = 0; it < fEgBx.size(); it++) {
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>* egLorentz =
          new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>();
      l1t::EGamma fakeEG(*egLorentz, fEgHwPt.at(it), fEgHwEta.at(it), fEgHwPhi.at(it), 0, fEgIso.at(it));
      egammas->push_back(fEgBx.at(it), fakeEG);
    }

    // Put Muons into Collections
    for (unsigned int it = 0; it < fMuBx.size(); it++) {
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>* muLorentz =
          new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>();
      l1t::Muon fakeMU(*muLorentz, fMuHwPt.at(it), fMuHwEta.at(it), fMuHwPhi.at(it), 4, 0, 0, fMuIso.at(it));
      muons->push_back(fMuBx.at(it), fakeMU);
    }

    // Put Taus into Collections
    for (unsigned int it = 0; it < fTauBx.size(); it++) {
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>* tauLorentz =
          new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>();
      l1t::Tau fakeTAU(*tauLorentz, fTauHwPt.at(it), fTauHwEta.at(it), fTauHwPhi.at(it), 0, fTauIso.at(it));
      taus->push_back(fTauBx.at(it), fakeTAU);
    }

    // Put Jets into Collections
    for (unsigned int it = 0; it < fJetBx.size(); it++) {
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>* jetLorentz =
          new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>();
      l1t::Jet fakeJET(*jetLorentz, fJetHwPt.at(it), fJetHwEta.at(it), fJetHwPhi.at(it), 0);
      jets->push_back(fJetBx.at(it), fakeJET);
    }

    // Put EtSums into Collections
    for (unsigned int it = 0; it < fEtSumBx.size(); it++) {
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>* etsumLorentz =
          new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>();
      l1t::EtSum fakeETSUM(
          *etsumLorentz, l1t::EtSum::EtSumType::kMissingEt, fEtSumHwPt.at(it), 0, fEtSumHwPhi.at(it), 0);
      etsums->push_back(fEtSumBx.at(it), fakeETSUM);
    }

    iEvent.put(std::move(egammas));
    iEvent.put(std::move(muons));
    iEvent.put(std::move(taus));
    iEvent.put(std::move(jets));
    iEvent.put(std::move(etsums));
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module ------------
  void FakeInputProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

}  // namespace l1t

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::FakeInputProducer);
