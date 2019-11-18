
/** \class PFRecoTauDiscriminationAgainstMuon2Container
 *
 * Compute tau Id. discriminator against muons.
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

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

using reco::tau::format_vint;

namespace {

  class PFRecoTauDiscriminationAgainstMuon2Container final : public PFTauDiscriminationProducerBaseForIDContainers {
    enum { kLoose, kMedium, kTight, kCustom };

  public:
    explicit PFRecoTauDiscriminationAgainstMuon2Container(const edm::ParameterSet& cfg)
        : PFTauDiscriminationProducerBaseForIDContainers(cfg), moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
      wpDefs_ = cfg.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
      // check content of discriminatorOption and add as enum to avoid string comparison per event
      for(std::vector<edm::ParameterSet>::iterator wpDefsEntry = wpDefs_.begin(); wpDefsEntry != wpDefs_.end(); ++wpDefsEntry){
        std::string discriminatorOption_string = wpDefsEntry->getParameter<std::string>("discriminatorOption");
        if (discriminatorOption_string == "loose")
          wpDefsEntry->addParameter<int>("discriminatorOptionEnum", kLoose);
        else if (discriminatorOption_string == "medium")
          wpDefsEntry->addParameter<int>("discriminatorOptionEnum", kMedium);
        else if (discriminatorOption_string == "tight")
          wpDefsEntry->addParameter<int>("discriminatorOptionEnum", kTight);
        else if (discriminatorOption_string == "custom")
          wpDefsEntry->addParameter<int>("discriminatorOptionEnum", kCustom);
        else
          throw edm::Exception(edm::errors::UnimplementedFeature)
              << " Invalid Configuration parameter 'discriminatorOption' = " << discriminatorOption_string << " !!\n";
      }
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
    ~PFRecoTauDiscriminationAgainstMuon2Container() override {}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    reco::PFSingleTauDiscriminatorContainer discriminate(const reco::PFTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::string moduleLabel_;
    std::vector<edm::ParameterSet> wpDefs_;
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

  std::atomic<unsigned int> PFRecoTauDiscriminationAgainstMuon2Container::numWarnings_{0};

  void PFRecoTauDiscriminationAgainstMuon2Container::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
    if (!srcMuons_.label().empty()) {
      evt.getByToken(Muons_token, muons_);
    }
  }

  reco::PFSingleTauDiscriminatorContainer PFRecoTauDiscriminationAgainstMuon2Container::discriminate(const reco::PFTauRef& pfTau) const {
    if (verbosity_) {
      edm::LogPrint("PFTauAgainstMuon2") << "<PFRecoTauDiscriminationAgainstMuon2Container::discriminate>:";
      edm::LogPrint("PFTauAgainstMuon2") << " moduleLabel = " << moduleLabel_;
      edm::LogPrint("PFTauAgainstMuon2") << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt()
                                         << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi();
    }

    std::vector<int> numMatchesDT(4);
    std::vector<int> numMatchesCSC(4);
    std::vector<int> numMatchesRPC(4);
    std::vector<int> numHitsDT(4);
    std::vector<int> numHitsCSC(4);
    std::vector<int> numHitsRPC(4);
    for (int iStation = 0; iStation < 4; ++iStation) {
      numMatchesDT[iStation] = 0;
      numMatchesCSC[iStation] = 0;
      numMatchesRPC[iStation] = 0;
      numHitsDT[iStation] = 0;
      numHitsCSC[iStation] = 0;
      numHitsRPC[iStation] = 0;
    }

    const reco::PFCandidatePtr& pfLeadChargedHadron = pfTau->leadPFChargedHadrCand();
    if (pfLeadChargedHadron.isNonnull()) {
      reco::MuonRef muonRef = pfLeadChargedHadron->muonRef();
      if (muonRef.isNonnull()) {
        if (verbosity_)
          edm::LogPrint("PFTauAgainstMuon2") << " has muonRef.";
        reco::tau::countMatches(*muonRef, numMatchesDT, numMatchesCSC, numMatchesRPC);
        reco::tau::countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }

    if (!srcMuons_.label().empty()) {
      size_t numMuons = muons_->size();
      for (size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon) {
        reco::MuonRef muon(muons_, idxMuon);
        if (verbosity_)
          edm::LogPrint("PFTauAgainstMuon2") << "muon #" << muon.key() << ": Pt = " << muon->pt()
                                             << ", eta = " << muon->eta() << ", phi = " << muon->phi();
        if (!(muon->pt() > minPtMatchedMuon_)) {
          if (verbosity_) {
            edm::LogPrint("PFTauAgainstMuon2") << " fails Pt cut --> skipping it.";
          }
          continue;
        }
        if (pfLeadChargedHadron.isNonnull()) {
          reco::MuonRef muonRef = pfLeadChargedHadron->muonRef();
          if (muonRef.isNonnull() && muon == pfLeadChargedHadron->muonRef()) {
            if (verbosity_) {
              edm::LogPrint("PFTauAgainstMuon2") << " matches muonRef of tau --> skipping it.";
            }
            continue;
          }
        }
        double dR = deltaR(muon->p4(), pfTau->p4());
        double dRmatch = dRmuonMatch_;
        if (dRmuonMatchLimitedToJetArea_) {
          double jetArea = 0.;
          if (pfTau->jetRef().isNonnull())
            jetArea = pfTau->jetRef()->jetArea();
          if (jetArea > 0.) {
            dRmatch = TMath::Min(dRmatch, TMath::Sqrt(jetArea / TMath::Pi()));
          } else {
            if (numWarnings_ < maxWarnings_) {
              edm::LogInfo("PFRecoTauDiscriminationAgainstMuon2Container::discriminate")
                  << "Jet associated to Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta()
                  << ", phi = " << pfTau->phi() << " has area = " << jetArea << " !!";
              ++numWarnings_;
            }
            dRmatch = 0.1;
          }
        }
        if (dR < dRmatch) {
          if (verbosity_)
            edm::LogPrint("PFTauAgainstMuon2") << " overlaps with tau, dR = " << dR;
          reco::tau::countMatches(*muon, numMatchesDT, numMatchesCSC, numMatchesRPC);
          reco::tau::countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
        }
      }
    }

    int numStationsWithMatches = 0;
    for (int iStation = 0; iStation < 4; ++iStation) {
      if (numMatchesDT[iStation] > 0 && !maskMatchesDT_[iStation])
        ++numStationsWithMatches;
      if (numMatchesCSC[iStation] > 0 && !maskMatchesCSC_[iStation])
        ++numStationsWithMatches;
      if (numMatchesRPC[iStation] > 0 && !maskMatchesRPC_[iStation])
        ++numStationsWithMatches;
    }

    int numLast2StationsWithHits = 0;
    for (int iStation = 2; iStation < 4; ++iStation) {
      if (numHitsDT[iStation] > 0 && !maskHitsDT_[iStation])
        ++numLast2StationsWithHits;
      if (numHitsCSC[iStation] > 0 && !maskHitsCSC_[iStation])
        ++numLast2StationsWithHits;
      if (numHitsRPC[iStation] > 0 && !maskHitsRPC_[iStation])
        ++numLast2StationsWithHits;
    }

    if (verbosity_) {
      edm::LogPrint("PFTauAgainstMuon2") << "numMatchesDT  = " << format_vint(numMatchesDT);
      edm::LogPrint("PFTauAgainstMuon2") << "numMatchesCSC = " << format_vint(numMatchesCSC);
      edm::LogPrint("PFTauAgainstMuon2") << "numMatchesRPC = " << format_vint(numMatchesRPC);
      edm::LogPrint("PFTauAgainstMuon2") << " --> numStationsWithMatches = " << numStationsWithMatches;
      edm::LogPrint("PFTauAgainstMuon2") << "numHitsDT  = " << format_vint(numHitsDT);
      edm::LogPrint("PFTauAgainstMuon2") << "numHitsCSC = " << format_vint(numHitsCSC);
      edm::LogPrint("PFTauAgainstMuon2") << "numHitsRPC = " << format_vint(numHitsRPC);
      edm::LogPrint("PFTauAgainstMuon2") << " --> numLast2StationsWithHits = " << numLast2StationsWithHits;
    }

    bool passesCaloMuonVeto = true;
    double energyECALplusHCAL;
    const reco::Track* leadTrack = nullptr;
    if (pfLeadChargedHadron.isNonnull()) {
      energyECALplusHCAL = pfLeadChargedHadron->ecalEnergy() + pfLeadChargedHadron->hcalEnergy();
      if (verbosity_) {
        if (pfLeadChargedHadron->trackRef().isNonnull()) {
          edm::LogPrint("PFTauAgainstMuon2")
              << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL
              << ", leadPFChargedHadronP = " << pfLeadChargedHadron->trackRef()->p();
        } else if (pfLeadChargedHadron->gsfTrackRef().isNonnull()) {
          edm::LogPrint("PFTauAgainstMuon2")
              << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL
              << ", leadPFChargedHadronP = " << pfLeadChargedHadron->gsfTrackRef()->p();
        }
      }
      if (pfLeadChargedHadron->trackRef().isNonnull())
        leadTrack = pfLeadChargedHadron->trackRef().get();
      else if (pfLeadChargedHadron->gsfTrackRef().isNonnull())
        leadTrack = pfLeadChargedHadron->gsfTrackRef().get();
    }
    reco::PFSingleTauDiscriminatorContainer result;
    for(std::vector<edm::ParameterSet>::const_iterator wpDefsEntry = wpDefs_.begin(); wpDefsEntry != wpDefs_.end(); ++wpDefsEntry){
      //extract WP parameters
      int discriminatorOption = wpDefsEntry->getParameter<int>("discriminatorOptionEnum");
      double hop = wpDefsEntry->getParameter<double>("HoPMin");
      int maxNumberOfMatches = wpDefsEntry->getParameter<int>("maxNumberOfMatches");
      bool doCaloMuonVeto = wpDefsEntry->getParameter<bool>("doCaloMuonVeto");
      int maxNumberOfHitsLast2Stations = wpDefsEntry->getParameter<int>("maxNumberOfHitsLast2Stations");
      
      if (pfLeadChargedHadron.isNonnull()) {
        if (pfTau->decayMode() == 0 && leadTrack && energyECALplusHCAL < (hop * leadTrack->p()))
          passesCaloMuonVeto = false;
      }

      bool discriminatorValue = false;
      if (discriminatorOption == kLoose && numStationsWithMatches <= maxNumberOfMatches)
        discriminatorValue = true;
      else if (discriminatorOption == kMedium && numStationsWithMatches <= maxNumberOfMatches &&
               numLast2StationsWithHits <= maxNumberOfHitsLast2Stations)
        discriminatorValue = true;
      else if (discriminatorOption == kTight && numStationsWithMatches <= maxNumberOfMatches &&
               numLast2StationsWithHits <= maxNumberOfHitsLast2Stations && passesCaloMuonVeto)
        discriminatorValue = true;
      else if (discriminatorOption == kCustom) {
        discriminatorValue = true;
        if (maxNumberOfMatches >= 0 && numStationsWithMatches > maxNumberOfMatches)
          discriminatorValue = false;
        if (maxNumberOfHitsLast2Stations >= 0 && numLast2StationsWithHits > maxNumberOfHitsLast2Stations)
          discriminatorValue = false;
        if (doCaloMuonVeto && !passesCaloMuonVeto)
          discriminatorValue = false;
      }
      result.workingPoints.push_back(discriminatorValue);
      if (verbosity_)
        edm::LogPrint("PFTauAgainstMuon2") << "--> returning discriminatorValue = " << discriminatorValue;
    }

    return result;
  }

}  // namespace

void PFRecoTauDiscriminationAgainstMuon2Container::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstMuon2Container
  edm::ParameterSetDescription desc;
  desc.add<std::vector<int>>("maskHitsRPC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
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
  desc.add<double>("dRmuonMatch", 0.3);
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("muons"));
  
  edm::ParameterSetDescription desc_wp;
  desc_wp.add<std::string>("IDname");
  desc_wp.add<std::string>("discriminatorOption");
  desc_wp.add<double>("HoPMin");
  desc_wp.add<int>("maxNumberOfMatches");
  desc_wp.add<bool>("doCaloMuonVeto");
  desc_wp.add<int>("maxNumberOfHitsLast2Stations");
  edm::ParameterSet pset_wp;
  pset_wp.addParameter<std::string>("pfRecoTauDiscriminationAgainstMuon2Container", "IDname");
  pset_wp.addParameter<std::string>("discriminatorOption", "loose");
  pset_wp.addParameter<double>("HoPMin", 0.2);
  pset_wp.addParameter<int>("maxNumberOfMatches", 0);
  pset_wp.addParameter<bool>("doCaloMuonVeto", false);
  pset_wp.addParameter<int>("maxNumberOfHitsLast2Stations", 0);
  std::vector<edm::ParameterSet> vpsd_wp;
  vpsd_wp.push_back(pset_wp);
  desc.addVPSet("IDWPdefinitions", desc_wp, vpsd_wp);
  
  descriptions.add("pfRecoTauDiscriminationAgainstMuon2Container", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon2Container);
