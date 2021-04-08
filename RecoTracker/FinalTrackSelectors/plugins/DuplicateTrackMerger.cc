/** \class DuplicateTrackMerger
 * 
 * selects pairs of tracks that should be single tracks
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackMerger.h"

#include "CommonTools/Utils/interface/DynArray.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <atomic>

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace reco;
namespace {
  class DuplicateTrackMerger final : public edm::stream::EDProducer<> {
  public:
    /// constructor
    explicit DuplicateTrackMerger(const edm::ParameterSet &iPara);
    /// destructor
    ~DuplicateTrackMerger() override;

    using CandidateToDuplicate = std::vector<std::pair<int, int>>;

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    /// produce one event
    void produce(edm::Event &, const edm::EventSetup &) override;

    bool checkForDisjointTracks(const reco::Track *t1, const reco::Track *t2, TSCPBuilderNoMaterial &tscpBuilder) const;
    bool checkForOverlappingTracks(
        const reco::Track *t1, const reco::Track *t2, unsigned int nvh1, unsigned int nvh2, double cosT) const;

  private:
    /// MVA discriminator
    const GBRForest *forest_;

    /// MVA weights file
    std::string dbFileName_;
    bool useForestFromDB_;
    std::string forestLabel_;

    std::string propagatorName_;
    std::string chi2EstimatorName_;

    /// track input collection
    edm::EDGetTokenT<reco::TrackCollection> trackSource_;
    /// minDeltaR3d cut value
    double minDeltaR3d2_;
    /// minBDTG cut value
    double minBDTG_;
    ///min pT cut value
    double minpT2_;
    ///min p cut value
    double minP_;
    ///max distance between two tracks at closest approach
    float maxDCA2_;
    ///max difference in phi between two tracks
    float maxDPhi_;
    ///max difference in Lambda between two tracks
    float maxDLambda_;
    ///max difference in transverse impact parameter between two tracks
    float maxDdxy_;
    ///max difference in longitudinal impact parameter between two tracks
    float maxDdsz_;
    ///max difference in q/p between two tracks
    float maxDQoP_;
    /// max number of hits for shorter track for the overlap check
    unsigned int overlapCheckMaxHits_;
    /// max number of missing layers for the overlap check
    unsigned int overlapCheckMaxMissingLayers_;
    /// min cosT for the overlap check
    double overlapCheckMinCosT_;

    const MagneticField *magfield_;
    const TrackerTopology *ttopo_;
    const TrackerGeometry *geom_;
    const Propagator *propagator_;
    const Chi2MeasurementEstimatorBase *chi2Estimator_;

    edm::ESGetToken<GBRForest, GBRWrapperRcd> forestToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
    edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> estimatorToken_;

    ///Merger
    TrackMerger merger_;

#ifdef EDM_ML_DEBUG
    bool debug_;
#endif
  };
}  // namespace

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TFile.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include "DuplicateTrackType.h"

namespace {

  void DuplicateTrackMerger::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source", edm::InputTag());
    desc.add<double>("minDeltaR3d", -4.0);
    desc.add<double>("minBDTG", -0.1);
    desc.add<double>("minpT", 0.2);
    desc.add<double>("minP", 0.4);
    desc.add<double>("maxDCA", 30.0);
    desc.add<double>("maxDPhi", 0.30);
    desc.add<double>("maxDLambda", 0.30);
    desc.add<double>("maxDdsz", 10.0);
    desc.add<double>("maxDdxy", 10.0);
    desc.add<double>("maxDQoP", 0.25);
    desc.add<unsigned>("overlapCheckMaxHits", 4);
    desc.add<unsigned>("overlapCheckMaxMissingLayers", 1);
    desc.add<double>("overlapCheckMinCosT", 0.99);
    desc.add<std::string>("forestLabel", "MVADuplicate");
    desc.add<std::string>("GBRForestFileName", "");
    desc.add<bool>("useInnermostState", true);
    desc.add<std::string>("ttrhBuilderName", "WithAngleAndTemplate");
    desc.add<std::string>("propagatorName", "PropagatorWithMaterial");
    desc.add<std::string>("chi2EstimatorName", "DuplicateTrackMergerChi2Est");
    descriptions.add("DuplicateTrackMerger", desc);
  }

  DuplicateTrackMerger::DuplicateTrackMerger(const edm::ParameterSet &iPara)
      : forest_(nullptr), merger_(iPara, consumesCollector()) {
    trackSource_ = consumes<reco::TrackCollection>(iPara.getParameter<edm::InputTag>("source"));
    minDeltaR3d2_ = iPara.getParameter<double>("minDeltaR3d");
    minDeltaR3d2_ *= std::abs(minDeltaR3d2_);
    minBDTG_ = iPara.getParameter<double>("minBDTG");
    minpT2_ = iPara.getParameter<double>("minpT");
    minpT2_ *= minpT2_;
    minP_ = iPara.getParameter<double>("minP");
    maxDCA2_ = iPara.getParameter<double>("maxDCA");
    maxDCA2_ *= maxDCA2_;
    maxDPhi_ = iPara.getParameter<double>("maxDPhi");
    maxDLambda_ = iPara.getParameter<double>("maxDLambda");
    maxDdsz_ = iPara.getParameter<double>("maxDdsz");
    maxDdxy_ = iPara.getParameter<double>("maxDdxy");
    maxDQoP_ = iPara.getParameter<double>("maxDQoP");
    overlapCheckMaxHits_ = iPara.getParameter<unsigned>("overlapCheckMaxHits");
    overlapCheckMaxMissingLayers_ = iPara.getParameter<unsigned>("overlapCheckMaxMissingLayers");
    overlapCheckMinCosT_ = iPara.getParameter<double>("overlapCheckMinCosT");

    produces<std::vector<TrackCandidate>>("candidates");
    produces<CandidateToDuplicate>("candidateMap");

    forestLabel_ = iPara.getParameter<std::string>("forestLabel");

    dbFileName_ = iPara.getParameter<std::string>("GBRForestFileName");
    useForestFromDB_ = dbFileName_.empty();

    propagatorName_ = iPara.getParameter<std::string>("propagatorName");
    chi2EstimatorName_ = iPara.getParameter<std::string>("chi2EstimatorName");
    if (useForestFromDB_) {
      forestToken_ = esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", forestLabel_));
    }
    magFieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
    trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
    geometryToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
    propagatorToken_ = esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", propagatorName_));
    estimatorToken_ =
        esConsumes<Chi2MeasurementEstimatorBase, TrackingComponentsRecord>(edm::ESInputTag("", chi2EstimatorName_));

    /*
  tmvaReader_ = new TMVA::Reader("!Color:Silent");
  tmvaReader_->AddVariable("ddsz",&tmva_ddsz_);
  tmvaReader_->AddVariable("ddxy",&tmva_ddxy_);
  tmvaReader_->AddVariable("dphi",&tmva_dphi_);
  tmvaReader_->AddVariable("dlambda",&tmva_dlambda_);
  tmvaReader_->AddVariable("dqoverp",&tmva_dqoverp_);
  tmvaReader_->AddVariable("delta3d_r",&tmva_d3dr_);
  tmvaReader_->AddVariable("delta3d_z",&tmva_d3dz_);
  tmvaReader_->AddVariable("outer_nMissingInner",&tmva_outer_nMissingInner_);
  tmvaReader_->AddVariable("inner_nMissingOuter",&tmva_inner_nMissingOuter_);
  tmvaReader_->BookMVA("BDTG",mvaFilePath);
  */
  }

  DuplicateTrackMerger::~DuplicateTrackMerger() {
    if (!useForestFromDB_)
      delete forest_;
  }

#ifdef VI_STAT
  struct Stat {
    Stat() : maxCos(1.1), nCand(0), nLoop0(0) {}
    ~Stat() { std::cout << "Stats " << nCand << ' ' << nLoop0 << ' ' << maxCos << std::endl; }
    std::atomic<float> maxCos;
    std::atomic<int> nCand, nLoop0;
  };
  Stat stat;
#endif

  template <typename T>
  void update_maximum(std::atomic<T> &maximum_value, T const &value) noexcept {
    T prev_value = maximum_value;
    while (prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value))
      ;
  }

  template <typename T>
  void update_minimum(std::atomic<T> &minimum_value, T const &value) noexcept {
    T prev_value = minimum_value;
    while (prev_value > value && !minimum_value.compare_exchange_weak(prev_value, value))
      ;
  }

  void DuplicateTrackMerger::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
    merger_.init(iSetup);

    if (!forest_) {
      if (useForestFromDB_) {
        edm::ESHandle<GBRForest> forestHandle = iSetup.getHandle(forestToken_);
        forest_ = forestHandle.product();
      } else {
        TFile gbrfile(dbFileName_.c_str());
        forest_ = dynamic_cast<const GBRForest *>(gbrfile.Get(forestLabel_.c_str()));
      }
    }

    //edm::Handle<edm::View<reco::Track> >handle;
    edm::Handle<reco::TrackCollection> handle;
    iEvent.getByToken(trackSource_, handle);
    auto const &tracks = *handle;

    edm::ESHandle<MagneticField> hmagfield = iSetup.getHandle(magFieldToken_);
    magfield_ = hmagfield.product();

    edm::ESHandle<TrackerTopology> httopo = iSetup.getHandle(trackerTopoToken_);
    ttopo_ = httopo.product();

    edm::ESHandle<TrackerGeometry> hgeom = iSetup.getHandle(geometryToken_);
    geom_ = hgeom.product();

    edm::ESHandle<Propagator> hpropagator = iSetup.getHandle(propagatorToken_);
    propagator_ = hpropagator.product();

    edm::ESHandle<Chi2MeasurementEstimatorBase> hestimator = iSetup.getHandle(estimatorToken_);
    chi2Estimator_ = hestimator.product();

    TSCPBuilderNoMaterial tscpBuilder;
    auto out_duplicateCandidates = std::make_unique<std::vector<TrackCandidate>>();

    auto out_candidateMap = std::make_unique<CandidateToDuplicate>();
    LogDebug("DuplicateTrackMerger") << "Number of tracks to be checked for merging: " << tracks.size();

#ifdef EDM_ML_DEBUG
    auto test = [&](const reco::Track *a, const reco::Track *b) {
      const auto ev = iEvent.id().event();
      const auto aOriAlgo = a->originalAlgo();
      const auto bOriAlgo = b->originalAlgo();
      const auto aSeed = a->seedRef().key();
      const auto bSeed = b->seedRef().key();
      return ((ev == 6903 && ((aOriAlgo == 23 && aSeed == 695 && bOriAlgo == 5 && bSeed == 652) ||
                              (aOriAlgo == 23 && aSeed == 400 && bOriAlgo == 7 && bSeed == 156) ||
                              (aOriAlgo == 4 && aSeed == 914 && bOriAlgo == 22 && bSeed == 503) ||
                              (aOriAlgo == 5 && aSeed == 809 && bOriAlgo == 4 && bSeed == 1030) ||
                              (aOriAlgo == 23 && aSeed == 749 && bOriAlgo == 5 && bSeed == 659) ||
                              (aOriAlgo == 4 && aSeed == 1053 && bOriAlgo == 23 && bSeed == 1035) ||
                              (aOriAlgo == 4 && aSeed == 810 && bOriAlgo == 5 && bSeed == 666) ||
                              (aOriAlgo == 4 && aSeed == 974 && bOriAlgo == 5 && bSeed == 778))) ||
              (ev == 6904 && ((aOriAlgo == 23 && aSeed == 526 && bOriAlgo == 5 && bSeed == 307) ||
                              (aOriAlgo == 4 && aSeed == 559 && bOriAlgo == 22 && bSeed == 472))) ||
              (ev == 6902 && ((aOriAlgo == 4 && aSeed == 750 && bOriAlgo == 22 && bSeed == 340) ||
                              (aOriAlgo == 4 && aSeed == 906 && bOriAlgo == 5 && bSeed == 609) ||
                              (aOriAlgo == 4 && aSeed == 724 && bOriAlgo == 5 && bSeed == 528) ||
                              (aOriAlgo == 4 && aSeed == 943 && bOriAlgo == 23 && bSeed == 739) ||
                              (aOriAlgo == 8 && aSeed == 2 && bOriAlgo == 9 && bSeed == 2282) ||
                              (aOriAlgo == 23 && aSeed == 827 && bOriAlgo == 5 && bSeed == 656) ||
                              (aOriAlgo == 22 && aSeed == 667 && bOriAlgo == 7 && bSeed == 516))));
    };
#endif

    // cache few "heavy to compute quantities
    int nTracks = 0;
    declareDynArray(const reco::Track *, tracks.size(), selTracks);
    declareDynArray(unsigned int, tracks.size(), nValidHits);
    declareDynArray(unsigned int, tracks.size(), oriIndex);
    for (auto i = 0U; i < tracks.size(); i++) {
      const reco::Track *rt1 = &tracks[i];
      if (rt1->innerMomentum().perp2() < minpT2_)
        continue;
      selTracks[nTracks] = rt1;
      nValidHits[nTracks] = rt1->numberOfValidHits();  // yes it is extremely heavy!
      oriIndex[nTracks] = i;
      ++nTracks;
    }

    for (int i = 0; i < nTracks; i++) {
      const reco::Track *rt1 = selTracks[i];
      for (int j = i + 1; j < nTracks; j++) {
        const reco::Track *rt2 = selTracks[j];

#ifdef EDM_ML_DEBUG
        debug_ = false;
        if (test(rt1, rt2) || test(rt2, rt1)) {
          debug_ = true;
          LogTrace("DuplicateTrackMerger")
              << "Track1 " << i << " originalAlgo " << rt1->originalAlgo() << " seed " << rt1->seedRef().key() << " pT "
              << std::sqrt(rt1->innerMomentum().perp2()) << " charge " << rt1->charge() << " outerPosition2 "
              << rt1->outerPosition().perp2() << "\n"
              << "Track2 " << j << " originalAlgo " << rt2->originalAlgo() << " seed " << rt2->seedRef().key() << " pT "
              << std::sqrt(rt2->innerMomentum().perp2()) << " charge " << rt2->charge() << " outerPosition2 "
              << rt2->outerPosition().perp2();
        }
#endif

        if (rt1->charge() != rt2->charge())
          continue;
        auto cosT = (*rt1).momentum().Dot((*rt2).momentum());  // not normalized!
        IfLogTrace(debug_, "DuplicateTrackMerger") << " cosT " << cosT;
        if (cosT < 0.)
          continue;
        cosT /= std::sqrt((*rt1).momentum().Mag2() * (*rt2).momentum().Mag2());

        const reco::Track *t1, *t2;
        unsigned int nhv1, nhv2;
        if (rt1->outerPosition().perp2() < rt2->outerPosition().perp2()) {
          t1 = rt1;
          nhv1 = nValidHits[i];
          t2 = rt2;
          nhv2 = nValidHits[j];
        } else {
          t1 = rt2;
          nhv1 = nValidHits[j];
          t2 = rt1;
          nhv2 = nValidHits[i];
        }
        auto deltaR3d2 = (t1->outerPosition() - t2->innerPosition()).mag2();

        if (t1->outerPosition().perp2() > t2->innerPosition().perp2())
          deltaR3d2 *= -1.0;
        IfLogTrace(debug_, "DuplicateTrackMerger")
            << " deltaR3d2 " << deltaR3d2 << " t1.outerPos2 " << t1->outerPosition().perp2() << " t2.innerPos2 "
            << t2->innerPosition().perp2();

        bool compatible = false;
        DuplicateTrackType duplType;
        if (deltaR3d2 >= minDeltaR3d2_) {
          compatible = checkForDisjointTracks(t1, t2, tscpBuilder);
          duplType = DuplicateTrackType::Disjoint;
        } else {
          compatible = checkForOverlappingTracks(t1, t2, nhv1, nhv2, cosT);
          duplType = DuplicateTrackType::Overlapping;
        }
        if (!compatible)
          continue;

        IfLogTrace(debug_, "DuplicateTrackMerger") << " marking as duplicates" << oriIndex[i] << ',' << oriIndex[j];
        out_duplicateCandidates->push_back(merger_.merge(*t1, *t2, duplType));
        out_candidateMap->emplace_back(oriIndex[i], oriIndex[j]);

#ifdef VI_STAT
        ++stat.nCand;
        //    auto cosT = float((*t1).momentum().unit().Dot((*t2).momentum().unit()));
        if (cosT > 0)
          update_minimum(stat.maxCos, float(cosT));
        else
          ++stat.nLoop0;
#endif
      }
    }
    iEvent.put(std::move(out_duplicateCandidates), "candidates");
    iEvent.put(std::move(out_candidateMap), "candidateMap");
  }

  bool DuplicateTrackMerger::checkForDisjointTracks(const reco::Track *t1,
                                                    const reco::Track *t2,
                                                    TSCPBuilderNoMaterial &tscpBuilder) const {
    IfLogTrace(debug_, "DuplicateTrackMerger") << " Checking for disjoint duplicates";

    FreeTrajectoryState fts1 = trajectoryStateTransform::outerFreeState(*t1, &*magfield_, false);
    FreeTrajectoryState fts2 = trajectoryStateTransform::innerFreeState(*t2, &*magfield_, false);
    GlobalPoint avgPoint((t1->outerPosition().x() + t2->innerPosition().x()) * 0.5,
                         (t1->outerPosition().y() + t2->innerPosition().y()) * 0.5,
                         (t1->outerPosition().z() + t2->innerPosition().z()) * 0.5);
    TrajectoryStateClosestToPoint TSCP1 = tscpBuilder(fts1, avgPoint);
    TrajectoryStateClosestToPoint TSCP2 = tscpBuilder(fts2, avgPoint);
    IfLogTrace(debug_, "DuplicateTrackMerger")
        << " TSCP1.isValid " << TSCP1.isValid() << " TSCP2.isValid " << TSCP2.isValid();
    if (!TSCP1.isValid())
      return false;
    if (!TSCP2.isValid())
      return false;

    const FreeTrajectoryState &ftsn1 = TSCP1.theState();
    const FreeTrajectoryState &ftsn2 = TSCP2.theState();

    IfLogTrace(debug_, "DuplicateTrackMerger") << " DCA2 " << (ftsn2.position() - ftsn1.position()).mag2();
    if ((ftsn2.position() - ftsn1.position()).mag2() > maxDCA2_)
      return false;

    auto qoverp1 = ftsn1.signedInverseMomentum();
    auto qoverp2 = ftsn2.signedInverseMomentum();
    float tmva_dqoverp_ = qoverp1 - qoverp2;
    IfLogTrace(debug_, "DuplicateTrackMerger") << " dqoverp " << tmva_dqoverp_;
    if (std::abs(tmva_dqoverp_) > maxDQoP_)
      return false;

    //auto pp = [&](TrajectoryStateClosestToPoint const & ts) { std::cout << ' ' << ts.perigeeParameters().vector()[0] << '/'  << ts.perigeeError().transverseCurvatureError();};
    //if(qoverp1*qoverp2 <0) { std::cout << "charge different " << qoverp1 <<',' << qoverp2; pp(TSCP1); pp(TSCP2); std::cout << std::endl;}

    // lambda = pi/2 - theta....  so l1-l2 == t2-t1
    float tmva_dlambda_ = ftsn2.momentum().theta() - ftsn1.momentum().theta();
    IfLogTrace(debug_, "DuplicateTrackMerger") << " dlambda " << tmva_dlambda_;
    if (std::abs(tmva_dlambda_) > maxDLambda_)
      return false;

    auto phi1 = ftsn1.momentum().phi();
    auto phi2 = ftsn2.momentum().phi();
    float tmva_dphi_ = phi1 - phi2;
    if (std::abs(tmva_dphi_) > float(M_PI))
      tmva_dphi_ = 2.f * float(M_PI) - std::abs(tmva_dphi_);
    IfLogTrace(debug_, "DuplicateTrackMerger") << " dphi " << tmva_dphi_;
    if (std::abs(tmva_dphi_) > maxDPhi_)
      return false;

    auto dxy1 =
        (-ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x()) / TSCP1.pt();
    auto dxy2 =
        (-ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x()) / TSCP2.pt();
    float tmva_ddxy_ = dxy1 - dxy2;
    IfLogTrace(debug_, "DuplicateTrackMerger") << " ddxy " << tmva_ddxy_;
    if (std::abs(tmva_ddxy_) > maxDdxy_)
      return false;

    auto dsz1 = ftsn1.position().z() * TSCP1.pt() / TSCP1.momentum().mag() -
                (ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x()) /
                    TSCP1.pt() * ftsn1.momentum().z() / ftsn1.momentum().mag();
    auto dsz2 = ftsn2.position().z() * TSCP2.pt() / TSCP2.momentum().mag() -
                (ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x()) /
                    TSCP2.pt() * ftsn2.momentum().z() / ftsn2.momentum().mag();
    float tmva_ddsz_ = dsz1 - dsz2;
    IfLogTrace(debug_, "DuplicateTrackMerger") << " ddsz " << tmva_ddsz_;
    if (std::abs(tmva_ddsz_) > maxDdsz_)
      return false;

    float tmva_d3dr_ = avgPoint.perp();
    float tmva_d3dz_ = avgPoint.z();
    float tmva_outer_nMissingInner_ = t2->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    float tmva_inner_nMissingOuter_ = t1->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);

    float gbrVals_[9];
    gbrVals_[0] = tmva_ddsz_;
    gbrVals_[1] = tmva_ddxy_;
    gbrVals_[2] = tmva_dphi_;
    gbrVals_[3] = tmva_dlambda_;
    gbrVals_[4] = tmva_dqoverp_;
    gbrVals_[5] = tmva_d3dr_;
    gbrVals_[6] = tmva_d3dz_;
    gbrVals_[7] = tmva_outer_nMissingInner_;
    gbrVals_[8] = tmva_inner_nMissingOuter_;

    auto mvaBDTG = forest_->GetClassifier(gbrVals_);
    IfLogTrace(debug_, "DuplicateTrackMerger") << " mvaBDTG " << mvaBDTG;
    if (mvaBDTG < minBDTG_)
      return false;

    //  std::cout << "to merge " << mvaBDTG << ' ' << std::copysign(std::sqrt(std::abs(deltaR3d2)),deltaR3d2) << ' ' << tmva_dphi_ << ' ' << TSCP1.pt() <<'/'<<TSCP2.pt() << std::endl;
    return true;
  }

  bool DuplicateTrackMerger::checkForOverlappingTracks(
      const reco::Track *t1, const reco::Track *t2, unsigned int nvh1, unsigned int nvh2, double cosT) const {
    // ensure t1 is the shorter track
    if (nvh2 < nvh1) {
      std::swap(t1, t2);
      std::swap(nvh1, nvh2);
    }

    IfLogTrace(debug_, "DuplicateTrackMerger")
        << " Checking for overlapping duplicates, cosT " << cosT << " t1 hits " << nvh1;
    if (cosT < overlapCheckMinCosT_)
      return false;
    if (nvh1 > overlapCheckMaxHits_)
      return false;

    // find the hit on the longer track on layer of the first hit of the shorter track
    auto findHitOnT2 = [&](const TrackingRecHit *hit1) {
      const auto hitDet = hit1->geographicalId().det();
      const auto hitSubdet = hit1->geographicalId().subdetId();
      const auto hitLayer = ttopo_->layer(hit1->geographicalId());
      return std::find_if(t2->recHitsBegin(), t2->recHitsEnd(), [&](const TrackingRecHit *hit2) {
        const auto &detId = hit2->geographicalId();
        return (detId.det() == hitDet && detId.subdetId() == hitSubdet && ttopo_->layer(detId) == hitLayer);
      });
    };

    auto t1HitIter = t1->recHitsBegin();
    if (!(*t1HitIter)->isValid()) {
      IfLogTrace(debug_, "DuplicateTrackMerger") << " first t1 hit invalid";
      return false;
    }
    auto t2HitIter = findHitOnT2(*t1HitIter);
    if (t2HitIter == t2->recHitsEnd()) {
      // if first hit not found, try with second
      // if that fails, then reject
      ++t1HitIter;
      assert(t1HitIter != t1->recHitsEnd());

      if (!(*t1HitIter)->isValid()) {
        IfLogTrace(debug_, "DuplicateTrackMerger") << " second t1 hit invalid";
        return false;
      }
      t2HitIter = findHitOnT2(*t1HitIter);
      if (t2HitIter == t2->recHitsEnd())
        return false;
    }
    IfLogTrace(debug_, "DuplicateTrackMerger")
        << " starting overlap check from t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
        << std::distance(t2->recHitsBegin(), t2HitIter);

    auto prevSubdet = (*t1HitIter)->geographicalId().subdetId();
    const TrajectoryStateOnSurface tsosInner = trajectoryStateTransform::innerStateOnSurface(*t2, *geom_, magfield_);

    ++t1HitIter;
    ++t2HitIter;
    unsigned int missedLayers = 0;
    while (t1HitIter != t1->recHitsEnd() && t2HitIter != t2->recHitsEnd()) {
      // in case of invalid hits, reject immediately
      if ((*t1HitIter)->getType() != TrackingRecHit::valid || trackerHitRTTI::isUndef(**t1HitIter) ||
          (*t2HitIter)->getType() != TrackingRecHit::valid || trackerHitRTTI::isUndef(**t2HitIter)) {
        IfLogTrace(debug_, "DuplicateTrackMerger")
            << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
            << std::distance(t2->recHitsBegin(), t2HitIter) << " either is invalid, types t1 "
            << (*t1HitIter)->getType() << " t2 " << (*t2HitIter)->getType();
        return false;
      }

      const auto &t1DetId = (*t1HitIter)->geographicalId();
      const auto &t2DetId = (*t2HitIter)->geographicalId();

      const auto t1Det = t1DetId.det();
      const auto t2Det = t2DetId.det();
      if (t1Det != DetId::Tracker || t2Det != DetId::Tracker) {
        IfLogTrace(debug_, "DuplicateTrackMerger") << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter)
                                                   << " t2 hit " << std::distance(t2->recHitsBegin(), t2HitIter)
                                                   << " either not from Tracker, dets t1 " << t1Det << " t2 " << t2Det;
        return false;
      }

      const auto t1Subdet = t1DetId.subdetId();
      const auto t1Layer = ttopo_->layer(t1DetId);

      // reject if hits have the same DetId but are different
      if (t1DetId == t2DetId) {
        if (!(*t1HitIter)->sharesInput(*t2HitIter, TrackingRecHit::all)) {
          IfLogTrace(debug_, "DuplicateTrackMerger")
              << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
              << std::distance(t2->recHitsBegin(), t2HitIter) << " same DetId (" << t1DetId.rawId()
              << ") but do not share all input";
          return false;
        }
      } else {
        const auto t2Subdet = t2DetId.subdetId();
        const auto t2Layer = ttopo_->layer(t2DetId);

        // reject if hits are on different layers
        if (t1Subdet != t2Subdet || t1Layer != t2Layer) {
          bool recovered = false;
          // but try to recover first by checking if either one has skipped over a layer
          if (t1Subdet == prevSubdet && t2Subdet != prevSubdet) {
            // t1 has a layer t2 doesn't
            ++t1HitIter;
            recovered = true;
          } else if (t1Subdet != prevSubdet && t2Subdet == prevSubdet) {
            // t2 has a layer t1 doesn't
            ++t2HitIter;
            recovered = true;
          } else if (t1Subdet == t2Subdet) {
            prevSubdet = t1Subdet;
            // same subdet, so layer must be different
            if (t2Layer > t1Layer) {
              // t1 has a layer t2 doesn't
              ++t1HitIter;
              recovered = true;
            } else if (t1Layer > t2Layer) {
              // t2 has a layer t1 doesn't
              ++t2HitIter;
              recovered = true;
            }
          }
          if (recovered) {
            ++missedLayers;
            if (missedLayers > overlapCheckMaxMissingLayers_) {
              IfLogTrace(debug_, "DuplicateTrackMerger") << " max number of missed layers exceeded";
              return false;
            }
            continue;
          }

          IfLogTrace(debug_, "DuplicateTrackMerger")
              << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
              << std::distance(t2->recHitsBegin(), t2HitIter) << " are on different layers (subdet, layer) t1 "
              << t1Subdet << "," << t1Layer << " t2 " << t2Subdet << "," << t2Layer;
          return false;
        }
        // reject if same layer (but not same hit) in non-pixel detector
        else if (t1Subdet != PixelSubdetector::PixelBarrel && t1Subdet != PixelSubdetector::PixelEndcap) {
          IfLogTrace(debug_, "DuplicateTrackMerger")
              << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
              << std::distance(t2->recHitsBegin(), t2HitIter) << " are on same layer, but in non-pixel detector (det "
              << t1Det << " subdet " << t1Subdet << " layer " << t1Layer << ")";
          return false;
        }
      }

      // Propagate longer track to the shorter track hit surface, check compatibility
      TrajectoryStateOnSurface tsosPropagated = propagator_->propagate(tsosInner, (*t1HitIter)->det()->surface());
      if (!tsosPropagated.isValid()) {  // reject immediately if TSOS is not valid
        IfLogTrace(debug_, "DuplicateTrackMerger")
            << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
            << std::distance(t2->recHitsBegin(), t2HitIter) << " TSOS not valid";
        return false;
      }
      auto passChi2Pair = chi2Estimator_->estimate(tsosPropagated, **t1HitIter);
      if (!passChi2Pair.first) {
        IfLogTrace(debug_, "DuplicateTrackMerger")
            << " t1 hit " << std::distance(t1->recHitsBegin(), t1HitIter) << " t2 hit "
            << std::distance(t2->recHitsBegin(), t2HitIter) << " hit chi2 compatibility failed with chi2 "
            << passChi2Pair.second;
        return false;
      }

      prevSubdet = t1Subdet;
      ++t1HitIter;
      ++t2HitIter;
    }
    if (t1HitIter != t1->recHitsEnd()) {
      IfLogTrace(debug_, "DuplicateTrackMerger") << " hits on t2 ended before hits on t1";
      return false;
    }

    IfLogTrace(debug_, "DuplicateTrackMerger") << " all hits on t2 are on layers whre t1 has also a hit";
    return true;
  }
}  // namespace

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DuplicateTrackMerger);
