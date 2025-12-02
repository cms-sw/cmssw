#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"

#include <TProfile.h>
#include <TProfile2D.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TEfficiency.h>

#include <vector>
#include <map>
#include <utility>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <array>
#include <initializer_list>
#include <sstream>

namespace trackerDTC {

  // stub resolution plots helper
  enum Resolution { R, Phi, Z, NumResolution };
  constexpr std::initializer_list<Resolution> AllResolution = {R, Phi, Z};
  constexpr auto NameResolution = {"R", "Phi", "Z"};
  inline std::string name(Resolution r) { return std::string(*(NameResolution.begin() + r)); }
  // max tracking efficiency plots helper
  enum Efficiency { Phi0, Pt, InvPt, D0, Z0, Eta, NumEfficiency };
  constexpr std::initializer_list<Efficiency> AllEfficiency = {Phi0, Pt, InvPt, D0, Z0, Eta};
  constexpr auto NameEfficiency = {"Phi0", "Pt", "InvPt", "D0", "Z0", "Eta"};
  inline std::string name(Efficiency e) { return std::string(*(NameEfficiency.begin() + e)); }

  /*! \class  trackerDTC::Analyzer
   *  \brief  Class to analyze hardware like structured TTStub Collection used by Track Trigger emulators, runs DTC stub emulation, plots performance & stub occupancy
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class Analyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    Analyzer(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // fills kinematic tp histograms
    void fill(const TPPtr& tpPtr, const std::vector<TH1F*> th1fs) const;
    // fill stub related histograms
    void fill(const tt::StreamStub& stream, int region, int channel, int& sum, TH2F* th2f);

    // ED input token of DTC stubs
    edm::EDGetTokenT<TTDTC> edGetTokenReco_;
    // ED input token of StubAssociation with selected TPs
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenMC_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // Associator token
    edm::ESGetToken<tt::Associator, tt::SetupRcd> esGetTokenAssociator_;
    // stores, calculates and provides run-time constants
    const tt::Setup* setup_;
    // enables analyze of TPs
    bool useMCTruth_;
    //
    int nEvents_ = 0;

    // Histograms

    TProfile* prof_;
    TProfile* profChannel_;
    TH1F* hisChannel_;
    TH2F* hisRZStubs_;
    TH2F* hisRZStubsLost_;
    TH2F* hisRZStubsEff_;
    std::vector<TH1F*> hisResolution_;
    std::vector<TProfile2D*> profResolution_;
    std::vector<TH1F*> hisEff_;
    std::vector<TH1F*> hisEffMC_;
    std::vector<TEfficiency*> eff_;

    // printout
    std::stringstream log_;
  };

  Analyzer::Analyzer(const edm::ParameterSet& iConfig) : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")) {
    usesResource("TFileService");
    // book in- and output ED products
    const auto& inputTagReco = iConfig.getParameter<edm::InputTag>("InputTagReco");
    edGetTokenReco_ = consumes(inputTagReco);
    if (useMCTruth_) {
      const auto& inputTagMC = iConfig.getParameter<edm::InputTag>("InputTagMC");
      edGetTokenMC_ = consumes(inputTagMC);
    }
    // book ES product
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenAssociator_ = esConsumes();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void Analyzer::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    // dtc
    dir = fs->mkdir("DTC");
    prof_ = dir.make<TProfile>("Counts", ";", 4, 0.5, 4.5);
    prof_->GetXaxis()->SetBinLabel(1, "Stubs");
    prof_->GetXaxis()->SetBinLabel(2, "Lost Stubs");
    prof_->GetXaxis()->SetBinLabel(3, "TPs");
    prof_->GetXaxis()->SetBinLabel(4, "total TPs");
    // channel occupancy
    constexpr int maxOcc = 180;
    const int numChannels = setup_->numDTCs() * setup_->numOverlappingRegions();
    hisChannel_ = dir.make<TH1F>("His Channel Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profChannel_ = dir.make<TProfile>("Prof Channel Occupancy", ";", numChannels, -.5, numChannels - .5);
    // max tracking efficiencies
    if (useMCTruth_) {
      constexpr std::array<int, NumEfficiency> binsEff{{9 * 8, 10, 16, 10, 30, 24}};
      constexpr std::array<std::pair<double, double>, NumEfficiency> rangesEff{
          {{-M_PI, M_PI}, {0., 100.}, {-1. / 3., 1. / 3.}, {-5., 5.}, {-15., 15.}, {-2.4, 2.4}}};
      dir = fs->mkdir("DTC/Effi");
      hisEff_.reserve(NumEfficiency);
      hisEffMC_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        hisEffMC_.emplace_back(
            dir.make<TH1F>(("HisTotalTP" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
      for (Efficiency e : AllEfficiency)
        hisEff_.emplace_back(dir.make<TH1F>(
            ("HisPassesTP" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
      eff_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        eff_.emplace_back(
            dir.make<TEfficiency>(("Eff" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
    }
    // lost stub fraction in r-z
    dir = fs->mkdir("DTC/Loss");
    constexpr int bins = 400;
    constexpr double maxZ = 300.;
    constexpr double maxR = 120.;
    hisRZStubs_ = dir.make<TH2F>("RZ Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZStubsLost_ = dir.make<TH2F>("RZ Stubs Lost", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZStubsEff_ = dir.make<TH2F>("RZ Stubs Eff", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    // stub parameter resolutions
    dir = fs->mkdir("DTC/Res");
    constexpr std::array<double, NumResolution> ranges{{.2, .0001, .5}};
    constexpr int binsHis = 100;
    hisResolution_.reserve(NumResolution);
    profResolution_.reserve(NumResolution);
    for (Resolution r : AllResolution) {
      hisResolution_.emplace_back(dir.make<TH1F>(("HisRes" + name(r)).c_str(), ";", binsHis, -ranges[r], ranges[r]));
      profResolution_.emplace_back(
          dir.make<TProfile2D>(("ProfRes" + name(r)).c_str(), ";;", bins, -maxZ, maxZ, bins, 0., maxR));
    }
  }

  void Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // read in dtc products
    const TTDTC& ttDTC = iEvent.get(edGetTokenReco_);
    // read in MCTruth
    tt::Associator associator = iSetup.getData(esGetTokenAssociator_);
    if (useMCTruth_) {
      associator.consume(iEvent.get(edGetTokenMC_));
      prof_->Fill(4, associator.numTPs());
      for (const auto& p : associator.getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffMC_);
    }
    // analyze dtc products and find still reconstrucable TrackingParticles
    std::set<TPPtr> tpPtrs;
    for (int region = 0; region < setup_->numRegions(); region++) {
      int nStubs(0);
      std::map<TPPtr, std::vector<TTStubRef>> mapTPsTTStubs;
      for (int channel = 0; channel < setup_->numDTCsPerTFP(); channel++) {
        const tt::StreamStub& stream = ttDTC.stream(region, channel);
        hisChannel_->Fill(stream.size());
        profChannel_->Fill(channel, stream.size());
        fill(stream, region, channel, nStubs, hisRZStubs_);
        if (!useMCTruth_)
          continue;
        for (const tt::FrameStub& frame : stream) {
          if (frame.first.isNull())
            continue;
          for (const TPPtr& tpPtr : associator.findTrackingParticlePtrs(frame.first)) {
            auto it = mapTPsTTStubs.find(tpPtr);
            if (it == mapTPsTTStubs.end()) {
              it = mapTPsTTStubs.emplace(tpPtr, std::vector<TTStubRef>()).first;
              it->second.reserve(associator.findTTStubRefs(tpPtr).size());
            }
            it->second.push_back(frame.first);
          }
        }
        for (const auto& p : mapTPsTTStubs)
          if (associator.reconstructable(p.second))
            tpPtrs.insert(p.first);
      }
      prof_->Fill(1, nStubs);
    }
    for (const TPPtr& tpPtr : tpPtrs)
      fill(tpPtr, hisEff_);
    prof_->Fill(3, tpPtrs.size());
    nEvents_++;
  }

  void Analyzer::endJob() {
    if (nEvents_ == 0)
      return;
    // create r-z stub fraction plot
    TH2F th2f("", ";;", 400, -300, 300., 400, 0., 120.);
    th2f.Add(hisRZStubsLost_);
    th2f.Add(hisRZStubs_);
    hisRZStubsEff_->Add(hisRZStubsLost_);
    hisRZStubsEff_->Divide(&th2f);
    // create efficieny plots
    if (useMCTruth_) {
      for (Efficiency e : AllEfficiency) {
        eff_[e]->SetPassedHistogram(*hisEff_[e], "f");
        eff_[e]->SetTotalHistogram(*hisEffMC_[e], "f");
      }
    }
    // printout DTC summary
    const double numStubs = prof_->GetBinContent(1);
    const double numTPs = prof_->GetBinContent(3);
    const double errStubs = prof_->GetBinError(1);
    const double totalTPs = prof_->GetBinContent(4);
    const double eff = numTPs / totalTPs;
    const double errEff = sqrt(eff * (1. - eff) / totalTPs / nEvents_);
    const std::vector<double> nums = {numStubs};
    const std::vector<double> errs = {errStubs};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "                         DTC SUMMARY                         " << std::endl;
    log_ << "number of stubs per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs) << errStubs
         << std::endl;
    log_ << "max tracking efficiency = " << std::setw(wNums) << eff << " +- " << std::setw(wErrs) << errEff
         << std::endl;
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

  // fills kinematic tp histograms
  void Analyzer::fill(const TPPtr& tpPtr, const std::vector<TH1F*> th1fs) const {
    const double s = sin(tpPtr->phi());
    const double c = cos(tpPtr->phi());
    const TrackingParticle::Point& v = tpPtr->vertex();
    const std::vector<double> x = {tpPtr->phi(),
                                   tpPtr->pt(),
                                   tpPtr->charge() / tpPtr->pt(),
                                   v.x() * s - v.y() * c,
                                   v.z() - (v.x() * c + v.y() * s) * sinh(tpPtr->eta()),
                                   tpPtr->eta()};
    for (Efficiency e : AllEfficiency)
      th1fs[e]->Fill(x[e]);
  }

  // fill stub related histograms
  void Analyzer::fill(const tt::StreamStub& stream, int region, int channel, int& sum, TH2F* th2f) {
    for (const tt::FrameStub& frame : stream) {
      if (frame.first.isNull())
        continue;
      sum++;
      const GlobalPoint& pos = setup_->stubPos(frame, region);
      const GlobalPoint& ttPos = setup_->stubPos(frame.first);
      const std::vector<double> resolutions = {
          ttPos.perp() - pos.perp(), tt::deltaPhi(ttPos.phi() - pos.phi()), ttPos.z() - pos.z()};
      for (Resolution r : AllResolution) {
        hisResolution_[r]->Fill(resolutions[r]);
        profResolution_[r]->Fill(ttPos.z(), ttPos.perp(), abs(resolutions[r]));
      }
      th2f->Fill(ttPos.z(), ttPos.perp());
    }
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
