#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/StubFE.h"
#include "L1Trigger/TrackerDTC/interface/StubGL.h"
#include "L1Trigger/TrackerDTC/interface/StubDTC.h"

#include <TProfile.h>
#include <TProfile2D.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TEfficiency.h>

#include <vector>
#include <array>
#include <map>
#include <set>
#include <utility>
#include <initializer_list>
#include <string>
#include <ios>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

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
  // stub resolution plots helper
  enum Module { Barrel2S, BarrelPSFlat, BarrelPSTilted, Disk2S, DiskPS, NumModule };
  constexpr std::initializer_list<Module> AllModule = {Barrel2S, BarrelPSFlat, BarrelPSTilted, Disk2S, DiskPS};
  constexpr auto NameModule = {"Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS"};
  inline std::string name(Module m) { return std::string(*(NameModule.begin() + m)); }

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
    void fill(const TPPtr&, const std::vector<TH1F*>) const;
    // fill stub related histograms
    void fill(const tt::StreamStub&, int, int, int&);
    // fill stub uncertainty histograms
    void fill(const TPPtr&, const tt::StreamStub&) const;
    // module type
    Module module(const SensorModule*) const;

    // ED input token of DTC stubs
    edm::EDGetTokenT<TTDTC> edGetTokenReco_;
    // ED input token of StubAssociation with selected TPs
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenMC_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // Associator token
    edm::ESGetToken<tt::Associator, trackerDTC::SetupRcd> esGetTokenAssociator_;
    // stores, calculates and provides run-time constants
    const Setup* setup_;
    // enables analyze of TPs
    bool useMCTruth_;
    //
    int nEvents_ = 0;
    //
    tt::Associator associator_;

    // Histograms

    TProfile* prof_;
    std::vector<std::vector<TH1F*>> hisResolution_;
    std::vector<TProfile2D*> profResolution_;
    std::vector<TH1F*> hisEff_;
    std::vector<TH1F*> hisEffMC_;
    std::vector<TEfficiency*> eff_;

    std::vector<TH1F*> hisClusterWidth_;
    TProfile2D* profClusterWidth_;

    std::vector<std::vector<TH1F*>> hisUncertainties_;
    std::vector<TProfile2D*> profUncertainties_;

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
    // stub parameter resolutions
    hisResolution_ = std::vector<std::vector<TH1F*>>(NumResolution);
    for (std::vector<TH1F*>& v : hisResolution_)
      v.reserve(NumModule);
    profResolution_.reserve(NumResolution);
    constexpr int bins = 400;
    constexpr double maxZ = 300.;
    constexpr double maxR = 120.;
    constexpr std::array<double, NumResolution> ranges{{.2, .0002, .5}};
    constexpr int binsHis = 100;
    for (Resolution r : AllResolution) {
      dir = fs->mkdir(("DTC/Res/" + name(r)).c_str());
      std::vector<TH1F*>& his = hisResolution_[r];
      for (Module m : AllModule)
        his.emplace_back(dir.make<TH1F>(("HisRes" + name(m)).c_str(), ";", binsHis, -ranges[r], ranges[r]));
      profResolution_.emplace_back(dir.make<TProfile2D>("Prof RZ Res", ";;", bins, -maxZ, maxZ, bins, 0., maxR));
    }
    // stub uncertenties
    hisUncertainties_ = std::vector<std::vector<TH1F*>>(NumResolution);
    for (std::vector<TH1F*>& v : hisUncertainties_)
      v.reserve(NumModule);
    profUncertainties_.reserve(NumResolution);
    for (Resolution r : AllResolution) {
      dir = fs->mkdir(("DTC/Uncertainty/" + name(r)).c_str());
      std::vector<TH1F*>& his = hisUncertainties_[r];
      for (Module m : AllModule)
        his.emplace_back(dir.make<TH1F>(("His d" + name(m)).c_str(), ";", 128, -5., 5.));
      profUncertainties_.emplace_back(dir.make<TProfile2D>("Prof RZ frac", ";;", bins, -maxZ, maxZ, bins, 0., maxR));
    }
    // cluster width
    dir = fs->mkdir("DTC/ClusterWidth");
    hisClusterWidth_.reserve(NumModule);
    for (Module m : AllModule)
      hisClusterWidth_.emplace_back(dir.make<TH1F>(("His " + name(m)).c_str(), ";", 5, 0, 5));
    profClusterWidth_ = dir.make<TProfile2D>("Prof RZ Cluster Width", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
  }

  void Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // read in dtc products
    const TTDTC& ttDTC = iEvent.get(edGetTokenReco_);
    // read in MCTruth
    associator_ = iSetup.getData(esGetTokenAssociator_);
    if (useMCTruth_) {
      associator_.consume(iEvent.get(edGetTokenMC_));
      prof_->Fill(4, associator_.numTPs());
      for (const auto& p : associator_.getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffMC_);
    }
    // analyze dtc products and find still reconstrucable TrackingParticles
    std::set<TPPtr> tpPtrs;
    for (int region = 0; region < setup_->sysNumRegion(); region++) {
      int nStubs(0);
      std::map<TPPtr, tt::StreamStub> mapTPsStreamStub;
      for (int channel = 0; channel < setup_->regNumDTC() * setup_->sysNumOverlap(); channel++) {
        const tt::StreamStub& stream = ttDTC.stream(region, channel);
        fill(stream, region, channel, nStubs);
        if (!useMCTruth_)
          continue;
        for (const tt::FrameStub& frame : stream) {
          if (frame.first.isNull())
            continue;
          for (const TPPtr& tpPtr : associator_.findTrackingParticlePtrs(frame.first)) {
            auto it = mapTPsStreamStub.find(tpPtr);
            if (it == mapTPsStreamStub.end()) {
              it = mapTPsStreamStub.emplace(tpPtr, tt::StreamStub()).first;
              it->second.reserve(associator_.findTTStubRefs(tpPtr).size());
            }
            it->second.push_back(frame);
          }
        }
        for (const auto& p : mapTPsStreamStub) {
          std::vector<TTStubRef> ttStubRefs;
          ttStubRefs.reserve(p.second.size());
          std::transform(p.second.begin(), p.second.end(), std::back_inserter(ttStubRefs), [](const tt::FrameStub f) {
            return f.first;
          });
          if (associator_.reconstructable(ttStubRefs)) {
            tpPtrs.insert(p.first);
            fill(p.first, p.second);
          }
        }
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
    log_ << "number of      stubs per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs) << errStubs
         << std::endl;
    log_ << "     max tracking efficiency = " << std::setw(wNums) << eff << " +- " << std::setw(wErrs) << errEff
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

  // fill stub uncertainty histograms
  void Analyzer::fill(const TPPtr& tpPtr, const tt::StreamStub& stream) const {
    const double inv2R = -tpPtr->charge() / tpPtr->pt() * setup_->sysInvPtToDphi();
    const double phi0 = tpPtr->phi();
    const double cot = tpPtr->tanl();
    const double z0 = tpPtr->z0();
    for (const tt::FrameStub& frame : stream) {
      const SensorModule* sm = setup_->sensorModule(frame.first);
      const GlobalPoint tt = setup_->stubPosTT(frame.first);
      const GlobalPoint dtc = setup_->stubPosDTC(frame, sm->dtcId() / setup_->regNumDTC());
      double dPhi = tt::deltaPhi(phi0 + tt.perp() * inv2R - tt.phi());
      double dZ = z0 + tt.perp() * cot - tt.z();
      double dR = tt.perp() - dtc.perp();
      Module m = module(sm);
      dPhi /= sm->dPhi(dtc.perp(), inv2R) / std::sqrt(12);
      dZ /= sm->dZ(cot) / std::sqrt(12);
      dR /= sm->dR() / std::sqrt(12);
      std::vector<double> d{{dR, dPhi, dZ}};
      for (Resolution r : AllResolution) {
        hisUncertainties_[r][m]->Fill(d[r]);
        if (std::abs(d[r]) < 3.)
          profUncertainties_[r]->Fill(tt.z(), tt.perp(), std::abs(d[r]));
      }
    }
  }

  // fill stub related histograms
  void Analyzer::fill(const tt::StreamStub& stream, int region, int channel, int& sum) {
    for (const tt::FrameStub& frame : stream) {
      if (frame.first.isNull())
        continue;
      sum++;
      const GlobalPoint& pos = setup_->stubPosDTC(frame, region);
      const GlobalPoint& ttPos = setup_->stubPosTT(frame.first);
      const SensorModule* sm = setup_->sensorModule(frame.first);
      Module m = module(sm);
      const std::vector<double> resolutions = {
          ttPos.perp() - pos.perp(), tt::deltaPhi(ttPos.phi() - pos.phi()), ttPos.z() - pos.z()};
      static const std::vector<double> limits = {1., 1.e-3, 1.};
      for (Resolution r : AllResolution) {
        hisResolution_[r][m]->Fill(resolutions[r]);
        profResolution_[r]->Fill(ttPos.z(), ttPos.perp(), std::abs(resolutions[r]));
        if (std::abs(resolutions[r]) > limits[r])
          throw cms::Exception("RuntimeError.");
      }
      const int width = frame.first->clusterRef(0)->findWidth();
      hisClusterWidth_[module(sm)]->Fill(width);
      profClusterWidth_->Fill(ttPos.z(), ttPos.perp(), width);
    }
  }

  // module type
  Module Analyzer::module(const SensorModule* sm) const {
    Module m;
    if (sm->barrel())
      if (sm->psModule())
        if (sm->tilted())
          m = Module::BarrelPSTilted;
        else
          m = Module::BarrelPSFlat;
      else
        m = Module::Barrel2S;
    else if (sm->psModule())
      m = Module::DiskPS;
    else
      m = Module::Disk2S;
    return m;
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
