// Author: Mariana Araujo
// Created: 2019-12-10
/*
Description:
HLT filter module to select events according to matching of central (jets) and PPS (RP tracks) kinematics

Implementation:
Matching can be done on the xi and/or mass+rapidity variables, using the do_xi and do_my booleans. If both are set to true, both matching conditions must be met
*/

// include files
#include "CondTools/RunInfo/interface/LHCInfoCombined.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"

// class declaration
//
class HLTPPSJetComparisonFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<float>> {
public:
  explicit HLTPPSJetComparisonFilter(const edm::ParameterSet &);
  ~HLTPPSJetComparisonFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  std::shared_ptr<float> globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                    const edm::EventSetup &) const override;

  void globalEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) const override;

  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcInfoPerLSToken_;
  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoPerFillToken_;

  const bool useNewLHCInfo_;

  const edm::InputTag jetInputTag_;  // Input tag identifying the jet track
  const edm::EDGetTokenT<reco::PFJetCollection> jet_token_;

  const edm::InputTag forwardProtonInputTag_;  // Input tag identifying the forward proton collection
  const edm::EDGetTokenT<std::vector<reco::ForwardProton>> recoProtonSingleRPToken_;

  const double maxDiffxi_;
  const double maxDiffm_;
  const double maxDiffy_;

  const unsigned int n_jets_;

  const bool do_xi_;
  const bool do_my_;
};

// fill descriptions
//
void HLTPPSJetComparisonFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("jetInputTag", edm::InputTag("hltAK4PFJetsCorrected"))
      ->setComment("input tag of the jet track collection");
  desc.add<edm::InputTag>("forwardProtonInputTag", edm::InputTag("ctppsProtons", "singleRP"))
      ->setComment("input tag of the forward proton collection");

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label used for LHCInfo");
  desc.add<std::string>("lhcInfoPerLSLabel", "")->setComment("label of the LHCInfoPerLS record");
  desc.add<std::string>("lhcInfoPerFillLabel", "")->setComment("label of the LHCInfoPerFill record");
  desc.add<bool>("useNewLHCInfo", true)->setComment("flag whether to use new LHCInfoPer* records or old LHCInfo");

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

std::shared_ptr<float> HLTPPSJetComparisonFilter::globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                                             const edm::EventSetup &iSetup) const {
  auto cache = std::make_shared<float>();

  LHCInfoCombined lhcInfoCombined(iSetup, lhcInfoPerLSToken_, lhcInfoPerFillToken_, lhcInfoToken_, useNewLHCInfo_);
  float sqs = 2. * lhcInfoCombined.energy;

  if (sqs == 0.f || !edm::isFinite(sqs)) {
    edm::LogError("HLTPPSJetComparisonFilter")
        << "LHC energy is zero (sqrt(s) = 0). All events in this IOV will be rejected.";
  }

  *cache = sqs;
  return cache;
}

void HLTPPSJetComparisonFilter::globalEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) const {}

HLTPPSJetComparisonFilter::HLTPPSJetComparisonFilter(const edm::ParameterSet &iConfig)
    : lhcInfoToken_(esConsumes<LHCInfo, LHCInfoRcd, edm::Transition::BeginLuminosityBlock>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
      lhcInfoPerLSToken_(esConsumes<LHCInfoPerLS, LHCInfoPerLSRcd, edm::Transition::BeginLuminosityBlock>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")))),
      lhcInfoPerFillToken_(esConsumes<LHCInfoPerFill, LHCInfoPerFillRcd, edm::Transition::BeginLuminosityBlock>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")))),
      useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")),

      jetInputTag_(iConfig.getParameter<edm::InputTag>("jetInputTag")),
      jet_token_(consumes<reco::PFJetCollection>(jetInputTag_)),

      forwardProtonInputTag_(iConfig.getParameter<edm::InputTag>("forwardProtonInputTag")),
      recoProtonSingleRPToken_(consumes<std::vector<reco::ForwardProton>>(forwardProtonInputTag_)),

      maxDiffxi_(iConfig.getParameter<double>("maxDiffxi")),
      maxDiffm_(iConfig.getParameter<double>("maxDiffm")),
      maxDiffy_(iConfig.getParameter<double>("maxDiffy")),

      n_jets_(iConfig.getParameter<unsigned int>("nJets")),

      do_xi_(iConfig.getParameter<bool>("do_xi")),
      do_my_(iConfig.getParameter<bool>("do_my")) {}

// member functions
//
bool HLTPPSJetComparisonFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  // get the cached value of the LHC energy from the cache
  const float sqs = *luminosityBlockCache(iEvent.getLuminosityBlock().index());

  // early return in case of not physical energy
  if (sqs == 0.f || !edm::isFinite(sqs))
    return false;

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
