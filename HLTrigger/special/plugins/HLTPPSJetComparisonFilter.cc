// Author: Mariana Araujo
// Created: 2019-12-10
/*
Description:
HLT filter module to select events according to matching of central (jets) and PPS (RP tracks) kinematics

Implementation:
Matching can be done on the xi and/or mass+rapidity variables, using the do_xi and do_my booleans. If both are set to true, both matching conditions must be met
*/

// include files
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "FWCore/Framework/interface/MakerMacros.h"

// class declaration
//
class HLTPPSJetComparisonFilter : public edm::global::EDFilter<> {
public:
  explicit HLTPPSJetComparisonFilter(const edm::ParameterSet &);
  ~HLTPPSJetComparisonFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  // ----------member data ---------------------------
  edm::ESGetToken<LHCInfo, LHCInfoRcd> const lhcInfoRcdToken_;

  edm::ParameterSet param_;

  edm::InputTag jetInputTag_;  // Input tag identifying the jet track
  edm::EDGetTokenT<reco::PFJetCollection> jet_token_;

  edm::InputTag forwardProtonInputTag_;  // Input tag identifying the forward proton collection
  edm::EDGetTokenT<std::vector<reco::ForwardProton>> recoProtonSingleRPToken_;

  std::string lhcInfoLabel_;

  double maxDiffxi_;
  double maxDiffm_;
  double maxDiffy_;

  unsigned int n_jets_;

  bool do_xi_;
  bool do_my_;
};

// fill descriptions
//
void HLTPPSJetComparisonFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("jetInputTag", edm::InputTag("hltAK4PFJetsCorrected"))
      ->setComment("input tag of the jet track collection");
  desc.add<edm::InputTag>("forwardProtonInputTag", edm::InputTag("ctppsProtons", "singleRP"))
      ->setComment("input tag of the forward proton collection");

  desc.add<std::string>("lhcInfoLabel", std::string(""))->setComment("label used for LHCInfo");

  desc.add<double>("maxDiffxi", 1.)
      ->setComment("maximum relative deviation of RP xi from dijet xi. Used with do_xi option");
  desc.add<double>("maxDiffm", 1.)
      ->setComment("maximum relative deviation of RP m from dijet m- Used with do_my option");
  desc.add<double>("maxDiffy", 1.)
      ->setComment("maximum absolute deviation of RP y from dijet y. Used with do_my option");

  desc.add<unsigned int>("nJets", 2)->setComment("number of jets to be used");

  desc.add<bool>("do_xi", true)->setComment("flag to require xi matching");
  desc.add<bool>("do_my", false)->setComment("flag to require m,y matching");

  descriptions.addWithDefaultLabel(desc);
  return;
}

// destructor and constructor
//
HLTPPSJetComparisonFilter::~HLTPPSJetComparisonFilter() = default;

HLTPPSJetComparisonFilter::HLTPPSJetComparisonFilter(const edm::ParameterSet &iConfig)
    : lhcInfoRcdToken_(esConsumes()),
      jetInputTag_(iConfig.getParameter<edm::InputTag>("jetInputTag")),
      jet_token_(consumes<reco::PFJetCollection>(jetInputTag_)),

      forwardProtonInputTag_(iConfig.getParameter<edm::InputTag>("forwardProtonInputTag")),
      recoProtonSingleRPToken_(consumes<std::vector<reco::ForwardProton>>(forwardProtonInputTag_)),

      lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),

      maxDiffxi_(iConfig.getParameter<double>("maxDiffxi")),
      maxDiffm_(iConfig.getParameter<double>("maxDiffm")),
      maxDiffy_(iConfig.getParameter<double>("maxDiffy")),

      n_jets_(iConfig.getParameter<unsigned int>("nJets")),

      do_xi_(iConfig.getParameter<bool>("do_xi")),
      do_my_(iConfig.getParameter<bool>("do_my")) {}

// member functions
//
bool HLTPPSJetComparisonFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  auto const &hLHCInfo = iSetup.getHandle(lhcInfoRcdToken_);
  float sqs = 2. * hLHCInfo->energy();  // get sqrt(s)

  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jet_token_, jets);  // get jet collection

  edm::Handle<std::vector<reco::ForwardProton>> recoSingleRPProtons;
  iEvent.getByToken(recoProtonSingleRPToken_, recoSingleRPProtons);  // get RP proton collection

  if (jets->size() < n_jets_)
    return false;  // test for nr jets

  if (do_xi_ && maxDiffxi_ > 0) {  // xi matching bloc

    float sum45 = 0, sum56 = 0;

    for (unsigned int i = 0; i < n_jets_; i++) {
      sum45 += (*jets)[i].energy() + (*jets)[i].pz();
      sum56 += (*jets)[i].energy() - (*jets)[i].pz();
    }

    const float xi45 = sum45 / sqs;  // get arm45 xi for n leading-pT jets
    const float xi56 = sum56 / sqs;  // get arm56 xi for n leading-pT jets

    float min45 = 1000., min56 = 1000.;

    for (const auto &proton : *recoSingleRPProtons)  // cycle over proton tracks
    {
      if (proton.validFit())  // Check that the track fit is valid
      {
        const auto &xi = proton.xi();  // Get the proton xi

        CTPPSDetId rpId(
            (*proton.contributingLocalTracks().begin())->rpId());  // get RP ID (rpId.arm() is 0 for 45 and 1 for 56)

        if (rpId.arm() == 0 && std::abs(xi - xi45) < min45)
          min45 = std::abs(xi - xi45);
        if (rpId.arm() == 1 && std::abs(xi - xi56) < min56)
          min56 = std::abs(xi - xi56);
      }
    }

    if (min56 / xi56 > maxDiffxi_ || min45 / xi45 > maxDiffxi_)
      return false;  // fail cond for xi matching
  }

  if (do_my_) {  // m, y matching bloc

    // get the mass and rap of the n jets
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>> j_sum;
    for (unsigned int i = 0; i < n_jets_; i++)
      j_sum = j_sum + (*jets)[i].p4();

    const auto &mjet = j_sum.M();
    const auto &yjet = j_sum.Rapidity();

    for (const auto &proton1 : *recoSingleRPProtons)  // cycle over first RP (only arm45)
    {
      if (proton1.validFit()) {
        CTPPSDetId rpId1(
            (*proton1.contributingLocalTracks().begin())->rpId());  // get RP ID (rpId.arm() is 0 for 45 and 1 for 56)
        if (rpId1.arm() == 0) {
          const auto &xi_45 = proton1.xi();

          for (const auto &proton2 : *recoSingleRPProtons)  // cycle over second RP (only arm56)
          {
            if (proton2.validFit()) {
              CTPPSDetId rpId2((*proton2.contributingLocalTracks().begin())->rpId());
              if (rpId2.arm() == 1) {
                const auto &xi_56 = proton2.xi();

                // m, y matching tests
                const auto &m = sqs * sqrt(xi_45 * xi_56);
                const auto &y = 0.5 * log(xi_45 / xi_56);
                if ((std::abs(m - mjet) / mjet < maxDiffm_ || maxDiffm_ <= 0) &&
                    (std::abs(y - yjet) < maxDiffy_ || maxDiffy_ <= 0))
                  return true;  // pass cond, immediately return true
              }
            }
          }
        }
      }
    }
    return false;  // fail cond for m,y matching (pass cond never met in cycle)
  }

  return true;  // if none of the fail conds are met, event has passed the trigger
}

DEFINE_FWK_MODULE(HLTPPSJetComparisonFilter);
