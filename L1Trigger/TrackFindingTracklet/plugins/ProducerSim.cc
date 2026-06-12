#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerSim
   *  \brief  simulation of Track Processing for extended track finding
   *  \author Thomas Schuh
   *  \date   2026, June
   */
  class ProducerSim : public edm::stream::EDProducer<> {
  public:
    explicit ProducerSim(const edm::ParameterSet&);
    ~ProducerSim() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    void beginRun(const edm::Run&, const edm::EventSetup&) override;

    // ED input token of TTTracks
    edm::EDGetTokenT<tt::TTTracks> edGetTokenTracks_;
    // ED output token of TTTracks
    edm::EDPutTokenT<tt::TTTracks> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // helper class to store configurations
    const Setup* setup_;
    //
    std::vector<int> nPer_;
    //
    int nPar_ = 5;
  };

  ProducerSim::ProducerSim(const edm::ParameterSet& iConfig) {
    const edm::InputTag& inputTag = iConfig.getParameter<edm::InputTag>("InputTagTracklet");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTTTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes(inputTag);
    edPutTokenTracks_ = produces(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
  }

  void ProducerSim::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // calc permutations for all found track sizes [4 - 7]
    auto fac = [](int n) {
      int f(1);
      for (int i = 1; i < n; i++)
        f *= i;
      return f;
    };
    nPer_.reserve(setup_->kfNumLayers() - setup_->kfMinLayers() + 1);
    for (int i = setup_->kfMinLayers(); i <= setup_->kfNumLayers(); i++)
      nPer_.push_back(fac(i) / fac(setup_->kfMinLayers()) / fac(i - setup_->kfMinLayers()));
  }

  void ProducerSim::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // read input
    edm::Handle<tt::TTTracks> handle;
    iEvent.getByToken(edGetTokenTracks_, handle);
    std::vector<TTTrackRef> ttTrackRefs;
    ttTrackRefs.reserve(handle->size());
    for (int iTrk = 0; iTrk < static_cast<int>(handle->size()); iTrk++)
      ttTrackRefs.emplace_back(handle, iTrk);
    // perform track multiplexinf
    const std::vector<int>& muxOrder = setup_->tmMuxOrder();
    auto order = [&muxOrder](const TTTrackRef& lhs, TTTrackRef& rhs) {
      const auto l = std::find(muxOrder.begin(), muxOrder.end(), lhs->trackSeedType());
      const auto r = std::find(muxOrder.begin(), muxOrder.end(), rhs->trackSeedType());
      return l < r;
    };
    std::sort(ttTrackRefs.begin(), ttTrackRefs.end(), order);
    // perform duplicate removal
    auto equalEnough = [this](const TTTrackRef& lhs, const TTTrackRef& rhs) {
      std::vector<TTStubRef> l = lhs->getStubRefs();
      std::vector<TTStubRef> r = rhs->getStubRefs();
      std::sort(l.begin(), l.end());
      std::sort(r.begin(), r.end());
      std::vector<TTStubRef> same;
      same.reserve(std::min(l.size(), r.size()));
      std::set_intersection(l.begin(), l.end(), r.begin(), r.end(), std::back_inserter(same));
      return static_cast<int>(same.size()) >= setup_->drMinIdenticalStubs();
    };
    std::vector<TTTrackRef*> ptrs;
    ptrs.reserve(ttTrackRefs.size());
    auto toPtr = [](TTTrackRef& ref) { return &ref; };
    std::transform(ttTrackRefs.begin(), ttTrackRefs.end(), std::back_inserter(ptrs), toPtr);
    for (int i = 0; i < static_cast<int>(ptrs.size()); i++) {
      TTTrackRef* master = ptrs[i];
      if (!master)
        continue;
      for (int j = i + 1; j < static_cast<int>(ptrs.size()); j++) {
        TTTrackRef*& slave = ptrs[j];
        if (!slave)
          continue;
        if (equalEnough(*master, *slave))
          slave = nullptr;
      }
    }
    for (int i = static_cast<int>(ptrs.size()) - 1; i >= 0; i--)
      if (!ptrs[i])
        ttTrackRefs.erase(std::next(ttTrackRefs.begin(), i));
    // perform KF
    tt::TTTracks ttTracks;
    ttTracks.reserve(ttTrackRefs.size());
    for (const TTTrackRef& ttTrackRef : ttTrackRefs) {
      const int iRegion = ttTrackRef->phiSector();
      const double phiR = iRegion * setup_->regRangePhiT();
      const double inv2R = -.5 * ttTrackRef->rInv();
      const double cot = ttTrackRef->tanL();
      const std::vector<TTStubRef>& ttStubRefs = ttTrackRef->getStubRefs();
      const int size = ttStubRefs.size();
      std::vector<std::vector<TTStubRef>> permutations;
      permutations.reserve(nPer_[size - setup_->kfMinLayers()]);
      for (int nStubs = setup_->kfMinLayers(); nStubs <= size; nStubs++) {
        permutations.emplace_back();
        std::vector<TTStubRef>& comb = permutations.back();
        comb.reserve(nStubs);
        // form all nStubs out of size combinations
        std::string bitmask(size, 1);
        bitmask.resize(setup_->kfMinLayers(), 0);
        do
          for (int i = 0; i < setup_->kfMinLayers(); ++i)
            if (bitmask[i])
              comb.push_back(ttStubRefs[i]);
        while (std::prev_permutation(bitmask.begin(), bitmask.end()));
      }
      ttTracks.emplace_back(0., 0., 0., 0., 0., 9.e3, 9.e3, 0., 0., 0., 0, nPar_, setup_->sysBField());
      TTTrack<Ref_Phase2TrackerDigi_>& ttTrack = ttTracks.back();
      ttTrack.setStubRefs(ttStubRefs);
      // fit all permutations
      for (const std::vector<TTStubRef>& permutation : permutations) {
        double x0(0.);
        double x1(0.);
        double x2(0.);
        double x3(0.);
        double x4(0.);
        double C00(9.e3);
        double C01(0.);
        double C11(9.e3);
        double C22(9.e3);
        double C23(0.);
        double C33(9.e3);
        double C44(9.e3);
        double C40(0.);
        double C41(0.);
        double chi20(0.);
        double chi21(0.);
        TTBV hitPattern(0, setup_->kfNumLayers());
        // add all stubs using KF update maths
        for (const TTStubRef& ttStubRef : permutation) {
          const GlobalPoint gp = setup_->stubPosTT(ttStubRef);
          const trackerDTC::SensorModule* sm = setup_->sensorModule(ttStubRef);
          const double m0 = tt::deltaPhi(gp.phi() - phiR);
          const double m1 = gp.z();
          const double v0 = std::pow(sm->dPhi(gp.perp(), inv2R), 2) / 12.;
          const double v1 = std::pow(sm->dZ(cot), 2) / 12.;
          const double H = gp.perp();
          const double r0 = m0 - x1 - x0 * H - x4 / H;
          const double r1 = m1 - x3 - x2 * H;
          const double S00 = C01 + H * C00 + C40 / H;
          const double S01 = C11 + H * C01 + C41 / H;
          const double S12 = C23 + H * C22;
          const double S13 = C33 + H * C23;
          const double S04 = C41 + H * C40 + C44 / H;
          const double R00 = v0 + S01 + H * S00 + S04 / H;
          const double R11 = v1 + S13 + H * S12;
          const double K00 = S00 / R00;
          const double K10 = S01 / R00;
          const double K21 = S12 / R11;
          const double K31 = S13 / R11;
          const double K40 = S04 / R00;
          x0 += r0 * K00;
          x1 += r0 * K10;
          x2 += r1 * K21;
          x3 += r1 * K31;
          x4 += r0 * K40;
          C00 -= S00 * K00;
          C01 -= S01 * K00;
          C11 -= S01 * K10;
          C22 -= S12 * K21;
          C23 -= S13 * K21;
          C33 -= S13 * K31;
          C44 -= S04 * K40;
          C40 -= S04 * K00;
          C41 -= S04 * K10;
          chi20 += r0 * r0 / R00;
          chi21 += r1 * r1 / R11;
          hitPattern.set(sm->layerIdReduced());
        }
        // TTTrack conversion
        TTTrack<Ref_Phase2TrackerDigi_> comb(-2. * x0,
                                             tt::deltaPhi(x1 + phiR),
                                             x2,
                                             x3,
                                             x4,
                                             chi20,
                                             chi21,
                                             0.,
                                             0.,
                                             0.,
                                             hitPattern.val(),
                                             nPar_,
                                             setup_->sysBField(),
                                             iRegion,
                                             ttTrackRef->etaSector(),
                                             0.,
                                             ttTrackRef->trackSeedType());
        // keep best combination
        if (comb.chi2Red() > ttTrack.chi2Red())
          continue;
        ttTrack = comb;
        ttTrack.setStubRefs(permutation);
      }
      // finish TTTrack
      ttTrack.setChi2BendRed(StubPtConsistency::getConsistency(
          ttTrack, setup_->trackerGeometry(), setup_->trackerTopology(), setup_->sysBField(), nPar_));
      ttTrack.setTrackWordBits();
    }
    // store products
    iEvent.emplace(edPutTokenTracks_, std::move(ttTracks));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerSim);
