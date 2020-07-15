
/** \class PFRecoTauDiscriminationAgainstMuon2
 *
 * Compute tau Id. discriminator against muons.
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstMuon2Helper.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"

#include <vector>
#include <string>
#include <iostream>
#include <atomic>

namespace {

  class PFRecoTauDiscriminationAgainstMuon2 final : public PFTauDiscriminationProducerBase {
  public:
    explicit PFRecoTauDiscriminationAgainstMuon2(const edm::ParameterSet& cfg)
        : PFTauDiscriminationProducerBase(cfg), moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
      std::string discriminatorOption_string = cfg.getParameter<std::string>("discriminatorOption");
      int discOption;
      if (discriminatorOption_string == "loose")
        discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kLoose;
      else if (discriminatorOption_string == "medium")
        discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kMedium;
      else if (discriminatorOption_string == "tight")
        discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kTight;
      else if (discriminatorOption_string == "custom")
        discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kCustom;
      else
        throw edm::Exception(edm::errors::UnimplementedFeature)
            << " Invalid Configuration parameter 'discriminatorOption' = " << discriminatorOption_string << " !!\n";
      wpDef_ = std::make_unique<PFRecoTauDiscriminationAgainstMuonConfigSet>(
          discOption,
          cfg.getParameter<double>("HoPMin"),
          cfg.getParameter<int>("maxNumberOfMatches"),
          cfg.getParameter<bool>("doCaloMuonVeto"),
          cfg.getParameter<int>("maxNumberOfHitsLast2Stations"));
      srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
      Muons_token = consumes<reco::MuonCollection>(srcMuons_);
      dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
      dRmuonMatchLimitedToJetArea_ = cfg.getParameter<bool>("dRmuonMatchLimitedToJetArea");
      minPtMatchedMuon_ = cfg.getParameter<double>("minPtMatchedMuon");
      typedef std::vector<int> vint;
      maskMatchesDT_ = cfg.getParameter<vint>("maskMatchesDT");
      maskMatchesCSC_ = cfg.getParameter<vint>("maskMatchesCSC");
      maskMatchesRPC_ = cfg.getParameter<vint>("maskMatchesRPC");
      maskHitsDT_ = cfg.getParameter<vint>("maskHitsDT");
      maskHitsCSC_ = cfg.getParameter<vint>("maskHitsCSC");
      maskHitsRPC_ = cfg.getParameter<vint>("maskHitsRPC");
      verbosity_ = cfg.getParameter<int>("verbosity");
    }
    ~PFRecoTauDiscriminationAgainstMuon2() override {}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    double discriminate(const reco::PFTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::string moduleLabel_;
    std::unique_ptr<PFRecoTauDiscriminationAgainstMuonConfigSet> wpDef_;
    edm::InputTag srcMuons_;
    edm::Handle<reco::MuonCollection> muons_;
    edm::EDGetTokenT<reco::MuonCollection> Muons_token;
    double dRmuonMatch_;
    bool dRmuonMatchLimitedToJetArea_;
    double minPtMatchedMuon_;
    std::vector<int> maskMatchesDT_;
    std::vector<int> maskMatchesCSC_;
    std::vector<int> maskMatchesRPC_;
    std::vector<int> maskHitsDT_;
    std::vector<int> maskHitsCSC_;
    std::vector<int> maskHitsRPC_;
    static std::atomic<unsigned int> numWarnings_;
    static constexpr unsigned int maxWarnings_ = 3;
    int verbosity_;
  };

  std::atomic<unsigned int> PFRecoTauDiscriminationAgainstMuon2::numWarnings_{0};

  void PFRecoTauDiscriminationAgainstMuon2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
    if (!srcMuons_.label().empty()) {
      evt.getByToken(Muons_token, muons_);
    }
  }

  double PFRecoTauDiscriminationAgainstMuon2::discriminate(const reco::PFTauRef& pfTau) const {
    const reco::PFCandidatePtr& pfCand = pfTau->leadPFChargedHadrCand();
    auto helper = PFRecoTauDiscriminationAgainstMuon2Helper(verbosity_,
                                                            moduleLabel_,
                                                            srcMuons_.label().empty(),
                                                            minPtMatchedMuon_,
                                                            dRmuonMatch_,
                                                            dRmuonMatchLimitedToJetArea_,
                                                            numWarnings_,
                                                            maxWarnings_,
                                                            maskMatchesDT_,
                                                            maskMatchesCSC_,
                                                            maskMatchesRPC_,
                                                            maskHitsDT_,
                                                            maskHitsCSC_,
                                                            maskHitsRPC_,
                                                            muons_,
                                                            pfTau,
                                                            pfCand);
    double discriminatorValue = helper.eval(*wpDef_, pfTau);
    if (verbosity_)
      edm::LogPrint("PFTauAgainstMuon2") << "--> returning discriminatorValue = " << discriminatorValue;
    return discriminatorValue;
  }

}  // namespace

void PFRecoTauDiscriminationAgainstMuon2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstMuon2
  edm::ParameterSetDescription desc;
  desc.add<std::vector<int>>("maskHitsRPC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<int>("maxNumberOfHitsLast2Stations", 0);
  desc.add<std::vector<int>>("maskMatchesRPC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<std::vector<int>>("maskMatchesCSC",
                             {
                                 1,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<std::vector<int>>("maskHitsCSC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  desc.add<int>("verbosity", 0);
  desc.add<std::vector<int>>("maskMatchesDT",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<double>("minPtMatchedMuon", 5.0);
  desc.add<bool>("dRmuonMatchLimitedToJetArea", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");              //, 0.5);
      psd1.add<edm::InputTag>("Producer");  //, edm::InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"));
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);  // optional with default?
    }
    // Prediscriminants can be
    // Prediscriminants = noPrediscriminants,
    // as in RecoTauTag/Configuration/python/HPSPFTaus_cff.py
    //
    // and the definition is:
    // RecoTauTag/RecoTau/python/TauDiscriminatorTools.py
    // # Require no prediscriminants
    // noPrediscriminants = cms.PSet(
    //       BooleanOperator = cms.string("and"),
    //       )
    // -- so this is the minimum required definition
    // otherwise it inserts the leadTrack with Producer = InpuTag(...) and does not find the corresponding output during the run
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<std::vector<int>>("maskHitsDT",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<double>("HoPMin", 0.2);
  desc.add<int>("maxNumberOfMatches", 0);
  desc.add<std::string>("discriminatorOption", "loose");
  desc.add<double>("dRmuonMatch", 0.3);
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("muons"));
  desc.add<bool>("doCaloMuonVeto", false);
  descriptions.add("pfRecoTauDiscriminationAgainstMuon2", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon2);
