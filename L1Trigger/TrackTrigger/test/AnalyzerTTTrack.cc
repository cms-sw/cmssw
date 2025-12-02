#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/L1Track.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"

#include <TProfile.h>
#include <TH1F.h>
#include <TEfficiency.h>

#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

namespace tt {

  /*! \class  tt::AnalyzerTTTrack
   *  \brief  Class to analyze TTTracks
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class AnalyzerTTTrack : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerTTTrack(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // plot helper
    std::vector<std::string> resolutions_ = {"Inv2R", "PT", "PhiT", "Phi0", "Cot", "ZT", "Z0", "D0"};
    std::vector<std::string> efficiencies_ = {"Inv2R", "PT", "Eta", "Z0", "D0"};
    std::vector<double> limitsR_ = {.001, 100., .01, .01, .2, 5., 5., 2.};
    std::vector<double> limitsE_ = {.001, 100., 2.4, 15., 10.};
    // ED input token of tracks
    edm::EDGetTokenT<std::vector<L1Track>> edGetTokenTracks_;
    // ED output token for stub association for fake rate
    edm::EDGetTokenT<StubAssociation> edGetTokenFake_;
    // ED output token for stub association duplicate rate
    edm::EDGetTokenT<StubAssociation> edGetTokenDup_;
    // ED output token for stub association for tracking efficiency
    edm::EDGetTokenT<StubAssociation> edGetTokenEff_;
    // Setup token
    edm::ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // Associator token
    edm::ESGetToken<Associator, SetupRcd> esGetTokenAssociator_;
    // enables analyze of TPs
    bool useMCTruth_;
    // input tag of TTTrack collection to be analyzed
    edm::InputTag inputTag_;
    // process name to be analyzed
    std::string name_;
    //
    bool looseMatching_;
    //
    int nEvents_ = 0;
    // Histograms
    TProfile* prof_;
    TH1F* hisLayer_;
    TH1F* hisStubs_;
    std::vector<TH1F*> hisRes_;
    std::vector<TProfile*> profRes_;
    std::vector<TH1F*> hisEffPassed_;
    std::vector<TH1F*> hisEffTotal_;
    std::vector<TEfficiency*> eff_;
    std::vector<TH1F*> hisChi2s_;
    // printout
    std::stringstream log_;
  };

  AnalyzerTTTrack::AnalyzerTTTrack(const edm::ParameterSet& iConfig)
      : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")),
        inputTag_(iConfig.getParameter<edm::InputTag>("InputTag")),
        name_(iConfig.getParameter<std::string>("Process")),
        looseMatching_(iConfig.getParameter<bool>("LooseMatching")),
        hisRes_(resolutions_.size()),
        profRes_(resolutions_.size()),
        hisEffPassed_(efficiencies_.size()),
        hisEffTotal_(efficiencies_.size()),
        eff_(efficiencies_.size()),
        hisChi2s_(2) {
    usesResource("TFileService");
    // book in- and output ED products
    edGetTokenTracks_ = consumes(inputTag_);
    if (useMCTruth_) {
      const std::string& labelMC = iConfig.getParameter<std::string>("LabelMC");
      const std::string& branchFake = iConfig.getParameter<std::string>("BranchFake");
      const std::string& branchDup = iConfig.getParameter<std::string>("BranchDup");
      const std::string& branchEff = iConfig.getParameter<std::string>("BranchEff");
      edGetTokenFake_ = consumes(edm::InputTag(labelMC, branchFake));
      edGetTokenDup_ = consumes(edm::InputTag(labelMC, branchDup));
      edGetTokenEff_ = consumes(edm::InputTag(labelMC, branchEff));
    }
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenAssociator_ = esConsumes();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerTTTrack::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir = fs->mkdir(name_);
    prof_ = dir.make<TProfile>("Counts", ";", 10, 0.5, 10.5);
    prof_->GetXaxis()->SetBinLabel(1, "Region Stubs");
    prof_->GetXaxis()->SetBinLabel(2, "Region Tracks");
    prof_->GetXaxis()->SetBinLabel(3, "All Tracks");
    prof_->GetXaxis()->SetBinLabel(4, "Matched to any Tracks");
    prof_->GetXaxis()->SetBinLabel(5, "Matched for Duplicates");
    prof_->GetXaxis()->SetBinLabel(6, "Found Any TPs");
    prof_->GetXaxis()->SetBinLabel(7, "Found for Duplicates TPs");
    prof_->GetXaxis()->SetBinLabel(8, "Found Selected TPs");
    prof_->GetXaxis()->SetBinLabel(9, "Found Perfect TPs");
    prof_->GetXaxis()->SetBinLabel(10, "All TPs");
    hisLayer_ = dir.make<TH1F>("Layer Occupancy", ";", 8, -0.5, 7.5);
    hisStubs_ = dir.make<TH1F>("Stubs per Track", ";", 8, .5, 8.5);
    // chi2s
    hisChi2s_[0] = dir.make<TH1F>("His Chi20", ";", 16, -.5, 15.5);
    hisChi2s_[1] = dir.make<TH1F>("His Chi21", ";", 16, -.5, 15.5);
    // resoultions
    dir = fs->mkdir(name_ + "/Res");
    for (int i = 0; i < static_cast<int>(resolutions_.size()); i++) {
      hisRes_[i] = dir.make<TH1F>(("His" + resolutions_[i]).c_str(), ";", 128, -limitsR_[i], limitsR_[i]);
      profRes_[i] = dir.make<TProfile>(("Prof" + resolutions_[i]).c_str(), ";", 32, 0, 2.4);
    }
    // Efficiencies
    dir = fs->mkdir(name_ + "/Eff");
    for (int i = 0; i < static_cast<int>(efficiencies_.size()); i++) {
      hisEffPassed_[i] = dir.make<TH1F>(("HisPassed" + efficiencies_[i]).c_str(), ";", 128, 0., limitsE_[i]);
      hisEffTotal_[i] = dir.make<TH1F>(("HisTotal" + efficiencies_[i]).c_str(), ";", 128, 0., limitsE_[i]);
      eff_[i] = dir.make<TEfficiency>(("Eff" + efficiencies_[i]).c_str(), ";", 128, 0., limitsE_[i]);
    }
  }

  void AnalyzerTTTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const Setup& setup = iSetup.getData(esGetTokenSetup_);
    auto fillEff = [&setup](const TPPtr& tpPtr, std::vector<TH1F*>& his) {
      his[0]->Fill(tpPtr->charge() / tpPtr->pt() * setup.invPtToDphi());
      his[1]->Fill(tpPtr->pt());
      his[2]->Fill(tpPtr->eta());
      his[3]->Fill(tpPtr->z0());
      his[4]->Fill(tpPtr->d0());
    };
    // read in tracks
    const std::vector<L1Track>& ttTracks = iEvent.get(edGetTokenTracks_);
    // read in MCTruth
    Associator forFake = iSetup.getData(esGetTokenAssociator_);
    Associator forDup = iSetup.getData(esGetTokenAssociator_);
    Associator forEff = iSetup.getData(esGetTokenAssociator_);
    if (useMCTruth_) {
      forFake.consume(iEvent.get(edGetTokenFake_));
      forDup.consume(iEvent.get(edGetTokenDup_));
      forEff.consume(iEvent.get(edGetTokenEff_));
      prof_->Fill(9, forEff.numTPs());
      for (const auto& p : forEff.getTrackingParticleToTTStubsMap())
        fillEff(p.first, hisEffTotal_);
    }
    // analyze and associate tracks with TrackingParticles
    std::set<TPPtr> tpPtrsDup;
    std::set<TPPtr> tpPtrsSelection;
    std::set<TPPtr> tpPtrsPerfect;
    const int allTracks = ttTracks.size();
    int allMatched(0);
    int allDuplicates(0);
    std::vector<int> regionStubs(setup.numRegions(), 0);
    std::vector<int> regionTracks(setup.numRegions(), 0);
    for (const L1Track& ttTrack : ttTracks) {
      const int region = ttTrack.phiSector();
      const std::vector<TTStubRef>& ttStubRefs = ttTrack.getStubRefs();
      regionTracks[region]++;
      regionStubs[region] += ttStubRefs.size();
      const TTBV hitPattern((int)ttTrack.hitPattern(), setup.numLayers());
      for (int layer : hitPattern.ids())
        hisLayer_->Fill(layer);
      hisStubs_->Fill(hitPattern.count());
      hisChi2s_[0]->Fill(ttTrack.getChi2RPhiBits());
      hisChi2s_[1]->Fill(ttTrack.getChi2RZBits());
      const std::vector<TPPtr>& any =
          looseMatching_ ? forFake.associate(ttStubRefs) : forFake.associateFinal(ttStubRefs);
      if (any.empty())
        continue;
      allMatched++;
      const std::vector<TPPtr>& dup = looseMatching_ ? forDup.associate(ttStubRefs) : forDup.associateFinal(ttStubRefs);
      if (dup.empty())
        continue;
      allDuplicates++;
      tpPtrsDup.insert(dup.begin(), dup.end());
      const std::vector<TPPtr>& select =
          looseMatching_ ? forEff.associate(ttStubRefs) : forEff.associateFinal(ttStubRefs);
      const std::vector<TPPtr>& perfect = forEff.associateFinal(ttStubRefs);
      tpPtrsSelection.insert(select.begin(), select.end());
      tpPtrsPerfect.insert(perfect.begin(), perfect.end());
      // calc resolutions
      const double tt_inv2R = -.5 * ttTrack.rInv();
      const double tt_pt = -setup.invPtToDphi() / tt_inv2R;
      const double tt_phi0 = ttTrack.phi();
      const double tt_phiT = tt::deltaPhi(tt_phi0 + setup.chosenRofPhi() * tt_inv2R);
      const double tt_cot = ttTrack.tanL();
      const double tt_z0 = ttTrack.z0();
      const double tt_zT = tt_z0 + setup.chosenRofZ() * tt_cot;
      const double tt_d0 = ttTrack.d0();
      for (const TPPtr& tpPtr : perfect) {
        const double eta = std::abs(tpPtr->eta());
        const double tp_inv2R = -tpPtr->charge() / tpPtr->pt() * setup.invPtToDphi();
        const double tp_pt = tpPtr->pt();
        const double tp_phi0 = tpPtr->phi();
        const double tp_phiT = tt::deltaPhi(tp_phi0 + setup.chosenRofPhi() * tp_inv2R);
        const double tp_cot = tpPtr->tanl();
        const double tp_z0 = tpPtr->z0();
        const double tp_zT = tp_z0 + setup.chosenRofZ() * tp_cot;
        const double tp_d0 = tpPtr->d0();
        const double inv2R = tp_inv2R - tt_inv2R;
        const double pt = tp_pt - tt_pt;
        const double phi0 = tt::deltaPhi(tp_phi0 - tt_phi0);
        const double phiT = tt::deltaPhi(tp_phiT - tt_phiT);
        const double cot = tp_cot - tt_cot;
        const double z0 = tp_z0 - tt_z0;
        const double zT = tp_zT - tt_zT;
        const double d0 = tp_d0 - tt_d0;
        int i(0);
        for (double d : {inv2R, pt, phiT, phi0, cot, z0, zT, d0}) {
          hisRes_[i]->Fill(d);
          profRes_[i++]->Fill(eta, std::abs(d));
        }
      }
    }
    for (int num : regionStubs)
      prof_->Fill(1, num);
    for (int num : regionTracks)
      prof_->Fill(2, num);
    prof_->Fill(3, allTracks);
    prof_->Fill(4, allMatched);
    prof_->Fill(5, allDuplicates);
    prof_->Fill(6, tpPtrsDup.size());
    prof_->Fill(7, tpPtrsSelection.size());
    prof_->Fill(8, tpPtrsPerfect.size());
    for (const TPPtr& tpPtr : tpPtrsPerfect)
      fillEff(tpPtr, hisEffPassed_);
    nEvents_++;
  }

  void AnalyzerTTTrack::endJob() {
    if (nEvents_ == 0)
      return;
    // effi
    for (int i = 0; i < static_cast<int>(efficiencies_.size()); i++) {
      eff_[i]->SetPassedHistogram(*hisEffPassed_[i], "f");
      eff_[i]->SetTotalHistogram(*hisEffTotal_[i], "f");
    }
    // printout summary
    const double regionStub = prof_->GetBinContent(1);
    const double regionTracks = prof_->GetBinContent(2);
    const double allTracks = prof_->GetBinContent(3);
    const double allMatched = prof_->GetBinContent(4);
    const double allDuplicates = prof_->GetBinContent(5);
    const double numDup = prof_->GetBinContent(6);
    const double numSelection = prof_->GetBinContent(7);
    const double numPerfect = prof_->GetBinContent(8);
    const double allTPs = prof_->GetBinContent(9);
    const double errStubs = prof_->GetBinError(1);
    const double errTracks = prof_->GetBinError(2);
    const double fracFake = (allTracks - allMatched) / allTracks;
    const double fracDup = (allDuplicates - numDup) / allTracks;
    const double effMax = numSelection / allTPs;
    const double effPerfect = numPerfect / allTPs;
    const double errEffMax = std::sqrt(effMax * (1. - effMax) / allTPs / nEvents_);
    const double errEffPerfect = std::sqrt(effPerfect * (1. - effPerfect) / allTPs / nEvents_);
    const std::vector<double> nums = {regionStub, regionTracks};
    const std::vector<double> errs = {errStubs, errTracks};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "                         " + name_ + "  SUMMARY                         " << std::endl;
    log_ << "   number of stubs  per TFP = " << std::setw(wNums) << regionStub << " +- " << std::setw(wErrs) << errStubs
         << std::endl;
    log_ << "   number of tracks per TFP = " << std::setw(wNums) << regionTracks << " +- " << std::setw(wErrs)
         << errTracks << std::endl;
    log_ << "current tracking efficiency = " << std::setw(wNums) << effPerfect << " +- " << std::setw(wErrs)
         << errEffPerfect << std::endl;
    log_ << "max     tracking efficiency = " << std::setw(wNums) << effMax << " +- " << std::setw(wErrs) << errEffMax
         << std::endl;
    log_ << "                  fake rate = " << std::setw(wNums) << fracFake << std::endl;
    log_ << "             duplicate rate = " << std::setw(wNums) << fracDup << std::endl;
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

}  // namespace tt

DEFINE_FWK_MODULE(tt::AnalyzerTTTrack);
