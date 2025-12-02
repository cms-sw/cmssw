#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <TProfile.h>
#include <TH1F.h>

#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace tt {

  /*! \class  tt::AnalyzerMC
   *  \brief  Class to analyze Tracking Particles, TTStubs and their association
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class AnalyzerMC : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    // Constructor
    explicit AnalyzerMC(const edm::ParameterSet& iConfig);
    // Destructor
    ~AnalyzerMC() override {}
    // Mandatory methods
    void beginJob() override {}
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // ED input token of TTStubs
    edm::EDGetTokenT<TTStubDetSetVec> edGetTokenTTStubs_;
    // ED input token of TTClusterAssociation
    edm::EDGetTokenT<TTClusterAssMap> edGetTokenTTClusterAssMap_;
    // ED input token of TPs
    edm::EDGetTokenT<TrackingParticleCollection> edGetTokenTPs_;
    // ED input token of TVs
    edm::EDGetTokenT<TrackingVertexCollection> edGetTokenTVs_;
    // ED input token of StubAssociation with reconstructable TPs
    edm::EDGetTokenT<StubAssociation> edGetTokenFake_;
    // ED input token of StubAssociation with selected TPs
    edm::EDGetTokenT<StubAssociation> edGetTokenEff_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // Histograms
    TProfile* prof_;
    // printout
    std::stringstream log_;
    //
    std::vector<TH1F*> hisZ_;
    std::vector<TH1F*> hisPhi_;
  };

  AnalyzerMC::AnalyzerMC(edm::ParameterSet const& iConfig) : hisZ_(5), hisPhi_(5) {
    usesResource("TFileService");
    const std::string& label = iConfig.getParameter<std::string>("StubAssociation");
    const std::string& branchFake = iConfig.getParameter<std::string>("BranchFake");
    const std::string& branchEff = iConfig.getParameter<std::string>("BranchEff");
    const edm::InputTag& inputTagTTStubs = iConfig.getParameter<edm::InputTag>("InputTagTTStubDetSetVec");
    const edm::InputTag& inputTagMC = iConfig.getParameter<edm::InputTag>("InputTagTTClusterAssMap");
    edGetTokenTTStubs_ = consumes(inputTagTTStubs);
    edGetTokenTTClusterAssMap_ = consumes(inputTagMC);
    edGetTokenFake_ = consumes(edm::InputTag(label, branchFake));
    edGetTokenEff_ = consumes(edm::InputTag(label, branchEff));
    // book ES product
    esGetTokenSetup_ = esConsumes();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerMC::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("MC");
    prof_ = dir.make<TProfile>("Counts", ";", 6, 0.5, 6.5);
    prof_->GetXaxis()->SetBinLabel(1, "Stubs");
    prof_->GetXaxis()->SetBinLabel(2, "matched Stubs");
    prof_->GetXaxis()->SetBinLabel(3, "reco Stubs");
    prof_->GetXaxis()->SetBinLabel(4, "any TPs");
    prof_->GetXaxis()->SetBinLabel(5, "reco TPs");
    prof_->GetXaxis()->SetBinLabel(6, "eff TPs");
    //
    dir = fs->mkdir("Stub uncertainties");
    const std::vector<std::string> names = {
        "pos charge pos d0", "pos charge neg d0", "neg charge pos d0", "neg charge neg d0", "all"};
    const double rPhi = 0.02;
    const double rZ = 2.;
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      hisZ_[i] = dir.make<TH1F>(("His stub z residual " + names[i]).c_str(), ";", 1024, -rZ / 2., rZ / 2.);
      hisPhi_[i] = dir.make<TH1F>(("His stub phi residual " + names[i]).c_str(), ";", 1024, -rPhi / 2., rPhi / 2.);
    }
  }

  void AnalyzerMC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // count stubs & matched stubs
    const TTStubDetSetVec& ttStubDetSetVec = iEvent.get(edGetTokenTTStubs_);
    const TTClusterAssMap& ttClusterAssMap = iEvent.get(edGetTokenTTClusterAssMap_);
    int nStubs(0);
    int nMatched(0);
    for (const auto& module : ttStubDetSetVec) {
      nStubs += module.size();
      for (const auto& ttStub : module) {
        bool matched(false);
        for (unsigned int iClus = 0; iClus < 2; iClus++)
          for (const TPPtr& tpPtr : ttClusterAssMap.findTrackingParticlePtrs(ttStub.clusterRef(iClus)))
            if (tpPtr.isNonnull())
              matched = true;
        if (matched)
          nMatched++;
      }
    }
    // get number of TPs
    const StubAssociation& forFake = iEvent.get(edGetTokenFake_);
    const StubAssociation& forEff = iEvent.get(edGetTokenEff_);
    const double numRegions = setup->numRegions();
    // stub uncertainties
    for (const auto& p : forEff.getTrackingParticleToTTStubsMap()) {
      const TPPtr& tpPtr = p.first;
      const double inv2R = -tpPtr->charge() / tpPtr->pt() * setup->invPtToDphi();
      const double phi0 = tpPtr->phi();
      const double cot = tpPtr->tanl();
      const double z0 = tpPtr->z0();
      const double d0 = -tpPtr->d0();
      const double R = .5 / inv2R;
      const double R0 = R + d0;
      int index = tpPtr->charge() < 0 ? 2 : 0;
      if (d0 < 0)
        index++;
      for (const TTStubRef& ttStubRef : p.second) {
        const GlobalPoint gp = setup->stubPos(ttStubRef);
        const double r = gp.perp();
        const double phi = phi0 + std::asin((r * r + R0 * R0 - R * R) / 2. / r / R0);
        const double z = z0 + std::abs(R) * cot * std::acos((R * R + R0 * R0 - r * r) / 2. / R / R0);
        const double dPhi = tt::deltaPhi(gp.phi() - phi);
        const double dZ = gp.z() - z;
        hisZ_[4]->Fill(dZ);
        hisPhi_[4]->Fill(dPhi);
        hisZ_[index]->Fill(dZ);
        hisPhi_[index]->Fill(dPhi);
      }
    }
    // store
    prof_->Fill(1, nStubs / numRegions);
    prof_->Fill(2, nMatched / numRegions);
    prof_->Fill(3, forFake.numStubs() / numRegions);
    prof_->Fill(4, ttClusterAssMap.getTrackingParticleToTTClustersMap().size() / numRegions);
    prof_->Fill(5, forFake.numTPs() / numRegions);
    prof_->Fill(6, forEff.numTPs() / numRegions);
  }

  // prints out Monte Carlo summary
  void AnalyzerMC::endJob() {
    const double numStubs = prof_->GetBinContent(1);
    const double numStubsMatched = prof_->GetBinContent(2);
    const double numStubsReco = prof_->GetBinContent(3);
    const double numTPsAny = prof_->GetBinContent(4);
    const double numTPsReco = prof_->GetBinContent(5);
    const double numTPsEff = prof_->GetBinContent(6);
    const double errStubs = prof_->GetBinError(1);
    const double errStubsMatched = prof_->GetBinError(2);
    const double errStubsReco = prof_->GetBinError(3);
    const double errTPsAny = prof_->GetBinError(4);
    const double errTPsReco = prof_->GetBinError(5);
    const double errTPsEff = prof_->GetBinError(6);
    const std::vector<double> nums = {numStubs, numStubsMatched, numStubsReco, numTPsAny, numTPsReco, numTPsEff};
    const std::vector<double> errs = {errStubs, errStubsMatched, errStubsReco, errTPsAny, errTPsReco, errTPsEff};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << std::endl;
    log_ << "                     Monte Carlo  SUMMARY                    " << std::endl;
    log_ << "number of stubs         per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs)
         << errStubs << std::endl;
    log_ << "number of matched stubs per TFP = " << std::setw(wNums) << numStubsMatched << " +- " << std::setw(wErrs)
         << errStubsMatched << std::endl;
    log_ << "number of reco stubs    per TFP = " << std::setw(wNums) << numStubsReco << " +- " << std::setw(wErrs)
         << errStubsReco << std::endl;
    log_ << "number of any TPs       per TFP = " << std::setw(wNums) << numTPsAny << " +- " << std::setw(wErrs)
         << errTPsAny << std::endl;
    log_ << "number of reco TPs      per TFP = " << std::setw(wNums) << numTPsReco << " +- " << std::setw(wErrs)
         << errTPsReco << std::endl;
    log_ << "number of TPs for eff   per TFP = " << std::setw(wNums) << numTPsEff << " +- " << std::setw(wErrs)
         << errTPsEff << std::endl;
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

}  // namespace tt

DEFINE_FWK_MODULE(tt::AnalyzerMC);
