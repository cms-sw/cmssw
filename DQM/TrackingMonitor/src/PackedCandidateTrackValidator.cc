#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/VecArray.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/Math/interface/liblogintpack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "boost/math/special_functions/sign.hpp"

#include <iomanip>

namespace {
  using dqm::reco::DQMStore;
  using dqm::reco::MonitorElement;

  template <typename T>
  void fillNoFlow(MonitorElement* me, T val) {
    auto h = me->getTH1();
    const auto xaxis = h->GetXaxis();
    if (val <= xaxis->GetXmin())
      h->AddBinContent(xaxis->GetFirst());
    else if (val >= xaxis->GetXmax())
      h->AddBinContent(xaxis->GetLast());
    else
      h->Fill(val);
  }

  class HitPatternPrinter {
  public:
    explicit HitPatternPrinter(const reco::Track& trk) : track(trk) {}

    void print(std::ostream& os) const {
      const reco::HitPattern& p = track.hitPattern();

      for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++i) {
        uint32_t hit = p.getHitPattern(reco::HitPattern::TRACK_HITS, i);

        detLayer(os, p, hit);
        if (p.missingHitFilter(hit)) {
          os << "(miss)";
        } else if (p.inactiveHitFilter(hit)) {
          os << "(inact)";
        } else if (p.badHitFilter(hit)) {
          os << "(bad)";
        }
        os << " ";
      }

      if (p.numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) > 0) {
        os << "lost inner ";

        for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); ++i) {
          uint32_t hit = p.getHitPattern(reco::HitPattern::MISSING_INNER_HITS, i);
          detLayer(os, p, hit);
          if (p.missingHitFilter(hit)) {
            os << "(miss)";
          } else if (p.inactiveHitFilter(hit)) {
            os << "(inact)";
          }
        }
      }
      if (p.numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) > 0) {
        os << "lost outer ";

        for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS); ++i) {
          uint32_t hit = p.getHitPattern(reco::HitPattern::MISSING_OUTER_HITS, i);
          detLayer(os, p, hit);
          if (p.missingHitFilter(hit)) {
            os << "(miss)";
          } else if (p.inactiveHitFilter(hit)) {
            os << "(inact)";
          }
        }
      }
    }

  private:
    static void detLayer(std::ostream& os, const reco::HitPattern& p, uint32_t hit) {
      if (p.pixelBarrelHitFilter(hit)) {
        os << "BPIX";
      } else if (p.pixelEndcapHitFilter(hit)) {
        os << "FPIX";
      } else if (p.stripTIBHitFilter(hit)) {
        os << "TIB";
      } else if (p.stripTIDHitFilter(hit)) {
        os << "TID";
      } else if (p.stripTOBHitFilter(hit)) {
        os << "TOB";
      } else if (p.stripTECHitFilter(hit)) {
        os << "TEC";
      }
      os << p.getLayer(hit);
    }

    const reco::Track& track;
  };

  std::ostream& operator<<(std::ostream& os, const HitPatternPrinter& hpp) {
    hpp.print(os);
    return os;
  }

  class TrackAlgoPrinter {
  public:
    explicit TrackAlgoPrinter(const reco::Track& trk) : track(trk) {}

    void print(std::ostream& os) const {
      edm::VecArray<reco::TrackBase::TrackAlgorithm, reco::TrackBase::algoSize> algos;
      for (int ialgo = 0; ialgo < reco::TrackBase::algoSize; ++ialgo) {
        auto algo = static_cast<reco::TrackBase::TrackAlgorithm>(ialgo);
        if (track.isAlgoInMask(algo)) {
          algos.push_back(algo);
        }
      }

      os << "algo " << reco::TrackBase::algoName(track.algo());
      if (track.originalAlgo() != track.algo())
        os << " originalAlgo " << reco::TrackBase::algoName(track.originalAlgo());
      if (algos.size() > 1) {
        os << " algoMask";
        for (auto algo : algos) {
          os << " " << reco::TrackBase::algoName(algo);
        }
      }
    }

  private:
    const reco::Track& track;
  };
  std::ostream& operator<<(std::ostream& os, const TrackAlgoPrinter& tap) {
    tap.print(os);
    return os;
  }

  double diffRelative(double a, double b) { return (a - b) / b; }

  class LogIntHelper {
  public:
    LogIntHelper(double lmin, double lmax) : lmin_(lmin), lmax_(lmax) {}

    class UnderOverflow {
    public:
      UnderOverflow(double largestValue, double smallestValue, std::function<double(double)> modifyUnpack)
          : unpackedLargestValue_(modifyUnpack ? modifyUnpack(largestValue) : largestValue),
            unpackedSmallestValue_(modifyUnpack ? modifyUnpack(smallestValue) : smallestValue) {}

      bool compatibleWithUnderflow(double value) const { return value == unpackedSmallestValue_; }
      void printNonOkUnderflow(std::ostream& os) const { os << " (not min " << unpackedSmallestValue_ << ")"; }

      bool compatibleWithOverflow(double value) const { return value == unpackedLargestValue_; }
      void printNonOkOverflow(std::ostream& os) const { os << " (not max " << unpackedLargestValue_ << ")"; }

    private:
      // narrow to float to compare apples to apples with values from
      // PackedCandidate (even though the final comparison is done in
      // double)
      const float unpackedLargestValue_;
      const float unpackedSmallestValue_;
    };

    static std::string maxName() { return "max"; }
    static std::string minName() { return "min"; }

    UnderOverflow underOverflowHelper(double value, std::function<double(double)> modifyUnpack) const {
      return UnderOverflow(
          largestValue(), value >= 0 ? smallestPositiveValue() : std::abs(smallestNegativeValue()), modifyUnpack);
    }

    double largestValue() const { return logintpack::unpack8log(127, lmin_, lmax_); }

    static bool wouldBeDenorm(double value) { return false; }

    // lessThan means closer to zero
    bool lessThanSmallestValue(double value) const {
      if (value >= 0)
        return value < smallestPositiveValue();
      else
        return value > smallestNegativeValue();
    }

    double smallestPositiveValue() const { return logintpack::unpack8log(logintpack::smallestPositive, lmin_, lmax_); }

    double smallestNegativeValue() const { return logintpack::unpack8log(logintpack::smallestNegative, lmin_, lmax_); }

  private:
    const double lmin_;
    const double lmax_;
  };
  class Float16Helper {
  public:
    class UnderOverflow {
    public:
      static void printNonOkUnderflow(std::ostream& os) { os << " (not 0)"; }
      static bool compatibleWithUnderflow(double value) { return value == 0.0; }
      static void printNonOkOverflow(std::ostream& os) { os << " (not inf)"; }
      static bool compatibleWithOverflow(double value) { return edm::isNotFinite(value); }
    };

    static std::string maxName() { return "inf"; }
    static std::string minName() { return "0"; }

    static UnderOverflow underOverflowHelper(double value, std::function<double(double)>) { return UnderOverflow(); }

    static double largestValue() { return MiniFloatConverter::max32RoundedToMax16(); }

    static bool wouldBeDenorm(double value) {
      const float valuef = static_cast<float>(value);
      return valuef >= MiniFloatConverter::denorm_min() && valuef < MiniFloatConverter::min();
    }

    static bool lessThanSmallestValue(double value) { return std::abs(value) < smallestValue(); }

    static double smallestValue() { return MiniFloatConverter::denorm_min(); }
  };

  enum class RangeStatus {
    inrange = 0,
    inrange_signflip = 1,
    denormal = 2,
    underflow_OK = 3,
    underflow_notOK = 4,
    overflow_OK = 5,
    overflow_notOK = 6
  };
  bool isInRange(RangeStatus status) { return status == RangeStatus::inrange || status == RangeStatus::denormal; }

  template <typename T>
  class PackedValueCheckResult {
  public:
    PackedValueCheckResult(RangeStatus status,
                           double diff,
                           double pcvalue,
                           double trackvalue,
                           double rangeMin,
                           double rangeMax,
                           const typename T::UnderOverflow& underOverflow)
        : diff_(diff),
          pcvalue_(pcvalue),
          trackvalue_(trackvalue),
          rangeMin_(rangeMin),
          rangeMax_(rangeMax),
          status_(status),
          underOverflow_(underOverflow) {}

    RangeStatus status() const { return status_; }
    double diff() const { return diff_; }

    bool outsideExpectedRange() const {
      if (status_ == RangeStatus::inrange)
        return diff_ < rangeMin_ || diff_ > rangeMax_;
      // denormal is considered as "in range" regardless of the expected range
      return status_ == RangeStatus::underflow_notOK || status_ == RangeStatus::overflow_notOK ||
             status_ == RangeStatus::inrange_signflip;
    }

    void print(std::ostream& os) const {
      if (outsideExpectedRange())
        os << "!! ";
      os << "(" << rangeMin_ << "," << rangeMax_ << ") ";

      os << diff_ << " " << pcvalue_;
      if (status_ == RangeStatus::underflow_OK || status_ == RangeStatus::underflow_notOK) {
        os << " (underflow) ";
        if (status_ == RangeStatus::underflow_notOK)
          underOverflow_.printNonOkUnderflow(os);
      } else if (status_ == RangeStatus::overflow_OK || status_ == RangeStatus::overflow_notOK) {
        os << " (overflow) ";
        if (status_ == RangeStatus::overflow_notOK)
          underOverflow_.printNonOkOverflow(os);
      } else if (status_ == RangeStatus::denormal)
        os << " (denormal)";
      os << " " << trackvalue_;
    }

  private:
    const double diff_;
    const double pcvalue_;
    const double trackvalue_;
    const double rangeMin_;
    const double rangeMax_;
    const RangeStatus status_;
    const typename T::UnderOverflow underOverflow_;
  };

  struct Range {
    Range(double mi, double ma) : min(mi), max(ma) {}
    const double min, max;
  };
  struct RangeAbs {
    RangeAbs(double val) : min(-val), max(val) {}
    const double min, max;
  };

  template <typename T>
  class PackedValueCheck {
  public:
    template <typename R, typename... Args>
    PackedValueCheck(const R& range, Args&&... args)
        : helper_(std::forward<Args>(args)...), rangeMin_(range.min), rangeMax_(range.max) {}

    void book(DQMStore::IBooker& iBooker,
              const std::string& name,
              const std::string& title,
              int nbins,
              double min,
              double max,
              int flow_nbins,
              double flow_min,
              double flow_max) {
      hInrange = iBooker.book1D(name, title, nbins, min, max);
      hUnderOverflowSign = iBooker.book1D(name + "UnderOverFlowSign",
                                          title + " with over- and underflow, and sign flip",
                                          flow_nbins,
                                          flow_min,
                                          flow_max);
      hStatus = iBooker.book1D(name + "Status", title + " status", 7, -0.5, 6.5);
      hStatus->setBinLabel(1, "In range");
      hStatus->setBinLabel(2, "In range, sign flip");
      hStatus->setBinLabel(3, "Denormal");
      hStatus->setBinLabel(4, "Underflow, PC is " + T::minName());
      hStatus->setBinLabel(5, "Underflow, PC is not " + T::minName());
      hStatus->setBinLabel(6, "Overflow, PC is " + T::maxName());
      hStatus->setBinLabel(7, "Overflow, PC is not " + T::maxName());
    }

    PackedValueCheckResult<T> fill(double pcvalue,
                                   double trackvalue,
                                   std::function<double(double)> modifyPack = std::function<double(double)>(),
                                   std::function<double(double)> modifyUnpack = std::function<double(double)>()) {
      const auto diff = diffRelative(pcvalue, trackvalue);

      const auto tmpSigned = modifyPack ? modifyPack(trackvalue) : trackvalue;
      const auto tmp = std::abs(tmpSigned);
      const auto underOverflow = helper_.underOverflowHelper(tmpSigned, modifyUnpack);
      RangeStatus status;
      if (tmp > helper_.largestValue()) {
        fillNoFlow(hUnderOverflowSign, diff);
        if (underOverflow.compatibleWithOverflow(std::abs(pcvalue))) {
          status = RangeStatus::overflow_OK;
        } else {
          status = RangeStatus::overflow_notOK;
        }
      } else if (helper_.lessThanSmallestValue(tmpSigned)) {
        fillNoFlow(hUnderOverflowSign, diff);
        if (underOverflow.compatibleWithUnderflow(std::abs(pcvalue))) {
          status = RangeStatus::underflow_OK;
        } else {
          status = RangeStatus::underflow_notOK;
        }
      } else {
        if (boost::math::sign(pcvalue) == boost::math::sign(trackvalue)) {
          if (T::wouldBeDenorm(tmp)) {
            status = RangeStatus::denormal;
          } else {
            status = RangeStatus::inrange;
          }
          fillNoFlow(hInrange, diff);
        } else {
          fillNoFlow(hUnderOverflowSign, diff);
          status = RangeStatus::inrange_signflip;
        }
      }
      hStatus->Fill(static_cast<int>(status));

      return PackedValueCheckResult<T>(status, diff, pcvalue, trackvalue, rangeMin_, rangeMax_, underOverflow);
    }

  private:
    const T helper_;
    const double rangeMin_;
    const double rangeMax_;

    MonitorElement* hInrange;
    MonitorElement* hUnderOverflowSign;
    MonitorElement* hStatus;
  };
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const PackedValueCheckResult<T>& res) {
    res.print(os);
    return os;
  }
}  // namespace

class PackedCandidateTrackValidator : public DQMEDAnalyzer {
public:
  PackedCandidateTrackValidator(const edm::ParameterSet& pset);
  ~PackedCandidateTrackValidator() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> trackToPackedCandidateToken_;

  std::string rootFolder_;
  bool debug_;

  enum {
    sf_AllTracks = 0,
    sf_AssociatedToPC = 1,
    sf_PCIsCharged = 2,
    sf_PCHasTrack = 3,
    sf_PCIsNotElectron = 4,
    sf_PCHasHits = 5,
    sf_PCNdofNot0 = 6,
    sf_NoMissingInnerHits = 7
  };
  MonitorElement* h_selectionFlow;

  MonitorElement* h_diffVx;
  MonitorElement* h_diffVy;
  MonitorElement* h_diffVz;

  MonitorElement* h_diffNormalizedChi2;
  MonitorElement* h_diffNdof;

  MonitorElement* h_diffCharge;
  MonitorElement* h_diffIsHighPurity;

  MonitorElement* h_diffPt;
  MonitorElement* h_diffEta;
  MonitorElement* h_diffPhi;
  PackedValueCheck<Float16Helper> h_diffDxyAssocPV;
  PackedValueCheck<Float16Helper> h_diffDzAssocPV;
  MonitorElement* h_diffDxyPV;
  MonitorElement* h_diffDzPV;

  MonitorElement* h_diffTrackDxyAssocPV;
  MonitorElement* h_diffTrackDzAssocPV;

  PackedValueCheck<LogIntHelper> h_diffCovQoverpQoverp;
  PackedValueCheck<LogIntHelper> h_diffCovLambdaLambda;
  PackedValueCheck<LogIntHelper> h_diffCovLambdaDsz;
  PackedValueCheck<LogIntHelper> h_diffCovPhiPhi;
  PackedValueCheck<LogIntHelper> h_diffCovPhiDxy;
  PackedValueCheck<Float16Helper> h_diffCovDxyDxy;
  PackedValueCheck<Float16Helper> h_diffCovDxyDsz;
  PackedValueCheck<Float16Helper> h_diffCovDszDsz;

  MonitorElement* h_diffDxyError;
  MonitorElement* h_diffDszError;
  MonitorElement* h_diffDzError;

  MonitorElement* h_diffTrackDxyError;
  MonitorElement* h_diffTrackDzError;

  MonitorElement* h_diffPtError;
  MonitorElement* h_diffEtaError;

  MonitorElement* h_diffNumberOfPixelLayers;
  MonitorElement* h_diffNumberOfStripLayers;
  MonitorElement* h_diffNumberOfPixelHits;
  MonitorElement* h_diffNumberOfHits;
  MonitorElement* h_diffLostInnerHits;

  MonitorElement* h_diffHitPatternPixelLayersWithMeasurement;
  MonitorElement* h_diffHitPatternTrackerLayersWithMeasurement;
  MonitorElement* h_diffHitPatternStripLayersWithMeasurement;
  MonitorElement* h_diffHitPatternNumberOfValidPixelHits;
  MonitorElement* h_diffHitPatternNumberOfValidHits;
  MonitorElement* h_diffHitPatternNumberOfLostInnerHits;
  MonitorElement* h_diffHitPatternHasValidHitInFirstPixelBarrel;

  MonitorElement* h_numberPixelLayersOverMax;
  MonitorElement* h_numberStripLayersOverMax;
  MonitorElement* h_numberLayersOverMax;
  MonitorElement* h_numberPixelHitsOverMax;
  MonitorElement* h_numberStripHitsOverMax;
  MonitorElement* h_numberHitsOverMax;
};

PackedCandidateTrackValidator::PackedCandidateTrackValidator(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<edm::View<reco::Track>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
      verticesToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
      trackToPackedCandidateToken_(consumes<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackToPackedCandidateAssociation"))),
      rootFolder_(iConfig.getUntrackedParameter<std::string>("rootFolder")),
      debug_(iConfig.getUntrackedParameter<bool>("debug")),
      h_diffDxyAssocPV(RangeAbs(0.001)),
      h_diffDzAssocPV(RangeAbs(0.001)),
      h_diffCovQoverpQoverp(Range(-1e-6, 0.13), -15, 0),  // despite of ceil in pack, there is rounding in double->float
      h_diffCovLambdaLambda(
          Range(-1e-6, 0.13), -20, -5),  // despite of ceil in pack, there is rounding in double->float
      h_diffCovLambdaDsz(RangeAbs(0.13), -17, -4),
      h_diffCovPhiPhi(RangeAbs(0.13), -15, 0),
      h_diffCovPhiDxy(RangeAbs(0.13), -17, -4),
      h_diffCovDxyDxy(RangeAbs(0.001)),
      h_diffCovDxyDsz(RangeAbs(0.001)),
      h_diffCovDszDsz(RangeAbs(0.001)) {}

PackedCandidateTrackValidator::~PackedCandidateTrackValidator() {}

void PackedCandidateTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("trackToPackedCandidateAssociation", edm::InputTag("packedPFCandidates"));
  desc.addUntracked<std::string>("rootFolder", "Tracking/PackedCandidate");
  desc.addUntracked<bool>("debug", false);

  descriptions.add("packedCandidateTrackValidator", desc);
}

void PackedCandidateTrackValidator::bookHistograms(DQMStore::IBooker& iBooker,
                                                   edm::Run const&,
                                                   edm::EventSetup const&) {
  iBooker.setCurrentFolder(rootFolder_);

  h_selectionFlow = iBooker.book1D("selectionFlow", "Track selection flow", 8, -0.5, 7.5);
  h_selectionFlow->setBinLabel(1, "All tracks");
  h_selectionFlow->setBinLabel(2, "Associated to PackedCandidate");
  h_selectionFlow->setBinLabel(3, "PC is charged"), h_selectionFlow->setBinLabel(4, "PC has track");
  h_selectionFlow->setBinLabel(5, "PC is not electron");
  h_selectionFlow->setBinLabel(6, "PC has hits");
  h_selectionFlow->setBinLabel(7, "PC ndof != 0");
  h_selectionFlow->setBinLabel(8, "Track: no missing inner hits");

  constexpr int diffBins = 50;

  h_diffVx =
      iBooker.book1D("diffVx", "PackedCandidate::bestTrack() - reco::Track in vx()", diffBins, -0.2, 0.2);  // not equal
  h_diffVy =
      iBooker.book1D("diffVy", "PackedCandidate::bestTrack() - reco::Track in vy()", diffBins, -0.2, 0.2);  // not equal
  h_diffVz =
      iBooker.book1D("diffVz", "PackedCandidate::bestTrack() - reco::Track in vz()", diffBins, -0.4, 0.4);  // not equal

  h_diffNormalizedChi2 = iBooker.book1D("diffNormalizedChi2",
                                        "PackedCandidate::bestTrack() - reco::Track in normalizedChi2()",
                                        30,
                                        -1.5,
                                        1.5);  // expected difference in -1...0
  h_diffNdof = iBooker.book1D(
      "diffNdof", "PackedCandidate::bestTrack() - reco::Track in ndof()", 33, -30.5, 2.5);  // to monitor the difference

  h_diffCharge = iBooker.book1D(
      "diffCharge", "PackedCandidate::bestTrack() - reco::Track in charge()", 5, -2.5, 2.5);  // expect equality
  h_diffIsHighPurity = iBooker.book1D("diffIsHighPurity",
                                      "PackedCandidate::bestTrack() - reco::Track in quality(highPurity)",
                                      3,
                                      -1.5,
                                      1.5);  // expect equality

  h_diffPt = iBooker.book1D("diffPt",
                            "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in pt()",
                            diffBins,
                            -1.1,
                            1.1);  // not equal, keep
  h_diffEta = iBooker.book1D(
      "diffEta", "PackedCandidate::bestTrack() - reco::Track in eta()", diffBins, -0.001, 0.001);  // not equal, keep
  h_diffPhi = iBooker.book1D("diffPhi",
                             "PackedCandidate::bestTrack() - reco::Track in phi()",
                             diffBins,
                             -0.0005,
                             0.0005);  // expect equality within precision

  h_diffDxyAssocPV.book(iBooker,
                        "diffDxyAssocPV",
                        "(PackedCandidate::dxy() - reco::Track::dxy(assocPV))/reco::Track",
                        40,
                        -0.001,
                        0.001,  // expect equality within precision
                        50,
                        -0.5,
                        0.5);
  h_diffDzAssocPV.book(iBooker,
                       "diffDzAssocPV",
                       "(PackedCandidate::dzAssociatedPV() - reco::Track::dz(assocPV))/reco::Track",
                       40,
                       -0.001,
                       0.001,  // expect equality within precision
                       50,
                       -0.5,
                       0.5);
  h_diffDxyPV = iBooker.book1D("diffDxyPV",
                               "(PackedCandidate::dxy(PV) - reco::Track::dxy(PV))/reco::Track",
                               diffBins,
                               -0.01,
                               0.01);  // expect equality within precision (worse than assocPV)
  h_diffDzPV = iBooker.book1D("diffDzPV",
                              "(PackedCandidate::dz(PV) - reco::Track::dz(PV))/reco::Track",
                              diffBins,
                              -0.01,
                              0.01);  // expect equality wihtin precision (worse than assocPV)
  h_diffTrackDxyAssocPV =
      iBooker.book1D("diffTrackDxyAssocPV",
                     "(PackedCandidate::bestTrack()::dxy(assocPV)) - reco::Track::dxy(assocPV))/reco::Track",
                     diffBins,
                     -0.01,
                     0.01);  // not equal
  h_diffTrackDzAssocPV =
      iBooker.book1D("diffTrackDzAssocPV",
                     "(PackedCandidate::bestTrack()::dz(assocPV)) - reco::Track::dz(assocPV))/reco::Track",
                     diffBins,
                     -0.01,
                     0.01);  // not equal

  h_diffCovQoverpQoverp.book(iBooker,
                             "diffCovQoverpQoverp",
                             "(PackedCandidate::bestTrack() - reco::Track)/reco::track in cov(qoverp, qoverp)",
                             40,
                             -0.05,
                             0.15,  // expect equality within precision (worst precision is exp(1/128*15) =~ 12 %
                             50,
                             -0.5,
                             0.5);
  h_diffCovLambdaLambda.book(
      iBooker,
      "diffCovLambdaLambda",
      "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(lambda, lambda)",
      40,
      -0.05,
      0.15,  // expect equality within precision worst precision is exp(1/128*(20-5)) =~ 12 % (multiplied by pt^2 in packing & unpacking)
      50,
      -0.5,
      0.5);
  h_diffCovLambdaDsz.book(iBooker,
                          "diffCovLambdaDsz",
                          "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(lambda, dsz)",
                          60,
                          -0.15,
                          0.15,  // expect equality within precision, worst precision is exp(1/128*(17-4) =~ 11 %
                          50,
                          -1,
                          1);
  h_diffCovPhiPhi.book(
      iBooker,
      "diffCovPhiPhi",
      "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(phi, phi)",
      40,
      -0.05,
      0.15,  // expect equality within precision worst precision is exp(1/128*(20-5)) =~ 12 % (multiplied by pt^2 in packing & unpacking)
      50,
      -0.5,
      0.5);
  h_diffCovPhiDxy.book(iBooker,
                       "diffCovPhiDxy",
                       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(phi, dxy)",
                       60,
                       -0.15,
                       0.15,  // expect equality within precision, wors precision is exp(1/128)*(17-4) =~ 11 %
                       50,
                       -1,
                       1);
  h_diffCovDxyDxy.book(iBooker,
                       "diffCovDxyDxy",
                       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(dxy, dxy)",
                       40,
                       -0.001,
                       0.001,
                       50,
                       -0.1,
                       0.1);
  h_diffCovDxyDsz.book(iBooker,
                       "diffCovDxyDsz",
                       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(dxy, dsz)",
                       40,
                       -0.001,
                       0.001,  // expect equality within precision
                       50,
                       -0.5,
                       0.5);
  h_diffCovDszDsz.book(iBooker,
                       "diffCovDszDsz",
                       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(dsz, dsz)",
                       40,
                       -0.001,
                       0.001,  // expect equality within precision
                       50,
                       -0.1,
                       0.1);

  h_diffDxyError = iBooker.book1D("diffDxyError",
                                  "(PackedCandidate::dxyError() - reco::Track::dxyError())/reco::Track",
                                  40,
                                  -0.001,
                                  0.001);  // expect equality within precision
  h_diffDszError = iBooker.book1D("diffDszError",
                                  "(PackedCandidate::dzError() - reco::Track::dszError())/reco::Track",
                                  40,
                                  -0.001,
                                  0.001);  // ideally, not equal, but for now they are
  h_diffDzError = iBooker.book1D("diffDzError",
                                 "(PackedCandidate::dzError() - reco::Track::dzError())/reco::Track",
                                 40,
                                 -0.001,
                                 0.001);  // expect equality within precision (not currently the case)

  h_diffTrackDxyError = iBooker.book1D("diffTrackDxyError",
                                       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dxyError()",
                                       40,
                                       -0.001,
                                       0.001);  // expect equality within precision
  h_diffTrackDzError = iBooker.book1D("diffTrackDzError",
                                      "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dzError()",
                                      40,
                                      -0.05,
                                      0.05);  // not equal

  h_diffPtError = iBooker.book1D("diffPtError",
                                 "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in ptError()",
                                 diffBins,
                                 -1.1,
                                 1.1);  // not equal
  h_diffEtaError = iBooker.book1D("diffEtaError",
                                  "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in etaError()",
                                  60,
                                  -0.15,
                                  0.15);  // not equal

  h_diffNumberOfPixelLayers = iBooker.book1D(
      "diffNumberOfPixelLayers",
      "PackedCandidate::pixelLayersWithMeasurement() - reco::Track::hitPattern::pixelLayersWithMeasurement()",
      5,
      -2.5,
      2.5);  // expect equality
  h_diffNumberOfStripLayers = iBooker.book1D(
      "diffNumberOfStripLayers",
      "PackedCandidate::stripLayersWithMeasurement() - reco::Track::hitPattern::stripLayersWithMeasurement()",
      5,
      -2.5,
      2.5);  // expect equality
  h_diffNumberOfPixelHits =
      iBooker.book1D("diffNumberOfPixelHits",
                     "PackedCandidate::numberOfPixelHits() - reco::Track::hitPattern::numberOfValidPixelHits()",
                     5,
                     -2.5,
                     2.5);  // expect equality
  h_diffNumberOfHits = iBooker.book1D("diffNumberOfHits",
                                      "PackedCandidate::numberHits() - reco::Track::hitPattern::numberOfValidHits()",
                                      5,
                                      -2.5,
                                      2.5);  // expect equality
  h_diffLostInnerHits =
      iBooker.book1D("diffLostInnerHits",
                     "PackedCandidate::lostInnerHits() - reco::Track::hitPattern::numberOfLostHits(MISSING_INNER_HITS)",
                     5,
                     -2.5,
                     2.5);  // expect equality

  h_diffHitPatternPixelLayersWithMeasurement =
      iBooker.book1D("diffHitPatternPixelLayersWithMeasurement",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::pixelLayersWithMeasurement()",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternStripLayersWithMeasurement =
      iBooker.book1D("diffHitPatternStripLayersWithMeasurement",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::stripLayersWithMeasurement()",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternTrackerLayersWithMeasurement =
      iBooker.book1D("diffHitPatternTrackerLayersWithMeasurement",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::trackerLayersWithMeasurement()",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternNumberOfValidPixelHits =
      iBooker.book1D("diffHitPatternNumberOfValidPixelHits",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidPixelHits()",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternNumberOfValidHits =
      iBooker.book1D("diffHitPatternNumberOfValidHits",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidHits()",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternNumberOfLostInnerHits =
      iBooker.book1D("diffHitPatternNumberOfLostPixelHits",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfLostHits(MISSING_INNER_HITS)",
                     13,
                     -10.5,
                     2.5);  // not equal
  h_diffHitPatternHasValidHitInFirstPixelBarrel =
      iBooker.book1D("diffHitPatternHasValidHitInFirstPixelBarrel",
                     "PackedCandidate::bestTrack() - reco::Track in hitPattern::hasValidHitInFirstPixelBarrel",
                     3,
                     -1.5,
                     1.5);  // expect equality

  h_numberPixelLayersOverMax = iBooker.book1D(
      "numberPixelLayersOverMax", "Number of pixel layers over the maximum of PackedCandidate", 10, 0, 10);
  h_numberStripLayersOverMax = iBooker.book1D(
      "numberStripLayersOverMax", "Number of strip layers over the maximum of PackedCandidate", 10, 0, 10);
  h_numberLayersOverMax =
      iBooker.book1D("numberLayersOverMax", "Number of layers over the maximum of PackedCandidate", 20, 0, 20);
  h_numberPixelHitsOverMax =
      iBooker.book1D("numberPixelHitsOverMax", "Number of pixel hits over the maximum of PackedCandidate", 10, 0, 10);
  h_numberStripHitsOverMax =
      iBooker.book1D("numberStripHitsOverMax", "Number of strip hits over the maximum of PackedCandidate", 10, 0, 10);
  h_numberHitsOverMax =
      iBooker.book1D("numberHitsOverMax", "Number of hits over the maximum of PackedCandidate", 20, 0, 20);
}

void PackedCandidateTrackValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Track>> htracks;
  iEvent.getByToken(tracksToken_, htracks);
  const auto& tracks = *htracks;

  edm::Handle<reco::VertexCollection> hvertices;
  iEvent.getByToken(verticesToken_, hvertices);
  const auto& vertices = *hvertices;

  if (vertices.empty())
    return;
  const reco::Vertex& pv = vertices[0];

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> hassoc;
  iEvent.getByToken(trackToPackedCandidateToken_, hassoc);
  const auto& trackToPackedCandidate = *hassoc;

  for (size_t i = 0; i < tracks.size(); ++i) {
    auto trackPtr = tracks.ptrAt(i);
    const reco::Track& track = *trackPtr;
    h_selectionFlow->Fill(sf_AllTracks);

    pat::PackedCandidateRef pcRef = trackToPackedCandidate[trackPtr];
    if (pcRef.isNull()) {
      continue;
    }
    h_selectionFlow->Fill(sf_AssociatedToPC);

    // Filter out neutral PackedCandidates, some of them may have track associated, and for those the charge comparison fails
    if (pcRef->charge() == 0) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCIsCharged);

    const reco::Track* trackPcPtr = pcRef->bestTrack();
    if (!trackPcPtr) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCHasTrack);

    // Filter out electrons to avoid comparisons to PackedCandidates with GsfTrack
    if (std::abs(pcRef->pdgId()) == 11) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCIsNotElectron);

    // Filter out PackedCandidate-tracks with no hits, as they won't have their details filled
    const reco::Track& trackPc = *trackPcPtr;
    if (trackPc.hitPattern().numberOfValidHits() == 0) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCHasHits);

    auto slimmedVertexRef = pcRef->vertexRef();
    const reco::Vertex& pcVertex = vertices[slimmedVertexRef.key()];

    fillNoFlow(h_diffVx, trackPc.vx() - track.vx());
    fillNoFlow(h_diffVy, trackPc.vy() - track.vy());
    fillNoFlow(h_diffVz, trackPc.vz() - track.vz());

    // PackedCandidate recalculates the ndof in unpacking as
    // (nhits+npixelhits-5), but some strip hits may have dimension 2.
    // If PackedCandidate has ndof=0, the resulting normalizedChi2
    // will be 0 too. Hence, the comparison makes sense only for those
    // PackedCandidates that have ndof != 0.
    double diffNormalizedChi2 = 0;
    if (trackPc.ndof() != 0) {
      h_selectionFlow->Fill(sf_PCNdofNot0);
      diffNormalizedChi2 = trackPc.normalizedChi2() - track.normalizedChi2();
      fillNoFlow(h_diffNormalizedChi2, diffNormalizedChi2);
    }
    fillNoFlow(h_diffNdof, trackPc.ndof() - track.ndof());

    auto diffCharge = trackPc.charge() - track.charge();
    fillNoFlow(h_diffCharge, diffCharge);
    int diffHP = static_cast<int>(trackPc.quality(reco::TrackBase::highPurity)) -
                 static_cast<int>(track.quality(reco::TrackBase::highPurity));
    fillNoFlow(h_diffIsHighPurity, diffHP);

    const auto diffPt = diffRelative(trackPc.pt(), track.pt());
    const auto diffPhi = reco::deltaPhi(trackPc.phi(), track.phi());
    fillNoFlow(h_diffPt, diffPt);
    fillNoFlow(h_diffEta, trackPc.eta() - track.eta());
    fillNoFlow(h_diffPhi, diffPhi);

    const auto diffDxyAssocPV =
        h_diffDxyAssocPV.fill(pcRef->dxy(), track.dxy(pcVertex.position()), [](double value) { return value * 100.; });
    const auto diffDzAssocPV = h_diffDzAssocPV.fill(
        pcRef->dzAssociatedPV(), track.dz(pcVertex.position()), [](double value) { return value * 100.; });
    const auto diffDxyPV = diffRelative(pcRef->dxy(pv.position()), track.dxy(pv.position()));
    const auto diffDzPV = diffRelative(pcRef->dz(pv.position()), track.dz(pv.position()));
    fillNoFlow(h_diffDxyPV, diffDxyPV);
    fillNoFlow(h_diffDzPV, diffDzPV);
    fillNoFlow(h_diffTrackDxyAssocPV, diffRelative(trackPc.dxy(pcVertex.position()), track.dxy(pcVertex.position())));
    fillNoFlow(h_diffTrackDzAssocPV, diffRelative(trackPc.dz(pcVertex.position()), track.dz(pcVertex.position())));

    auto fillCov1 = [&](auto& hlp, const int i, const int j) {
      return hlp.fill(trackPc.covariance(i, j), track.covariance(i, j));
    };
    auto fillCov2 = [&](auto& hlp, const int i, const int j, std::function<double(double)> modifyPack) {
      return hlp.fill(trackPc.covariance(i, j), track.covariance(i, j), modifyPack);
    };
    auto fillCov3 = [&](auto& hlp,
                        const int i,
                        const int j,
                        std::function<double(double)> modifyPack,
                        std::function<double(double)> modifyUnpack) {
      return hlp.fill(trackPc.covariance(i, j), track.covariance(i, j), modifyPack, modifyUnpack);
    };

    const auto pcPt = pcRef->pt();
    const auto diffCovQoverpQoverp = fillCov3(
        h_diffCovQoverpQoverp,
        reco::TrackBase::i_qoverp,
        reco::TrackBase::i_qoverp,
        [=](double val) { return val * pcPt * pcPt; },
        [=](double val) { return val / pcPt / pcPt; });
    const auto diffCovLambdaLambda =
        fillCov1(h_diffCovLambdaLambda, reco::TrackBase::i_lambda, reco::TrackBase::i_lambda);
    const auto diffCovLambdaDsz = fillCov1(h_diffCovLambdaDsz, reco::TrackBase::i_lambda, reco::TrackBase::i_dsz);
    const auto diffCovPhiPhi = fillCov3(
        h_diffCovPhiPhi,
        reco::TrackBase::i_phi,
        reco::TrackBase::i_phi,
        [=](double val) { return val * pcPt * pcPt; },
        [=](double val) { return val / pcPt / pcPt; });
    const auto diffCovPhiDxy = fillCov1(h_diffCovPhiDxy, reco::TrackBase::i_phi, reco::TrackBase::i_dxy);
    const auto diffCovDxyDxy = fillCov2(
        h_diffCovDxyDxy, reco::TrackBase::i_dxy, reco::TrackBase::i_dxy, [](double value) { return value * 10000.; });
    const auto diffCovDxyDsz = fillCov2(
        h_diffCovDxyDsz, reco::TrackBase::i_dxy, reco::TrackBase::i_dsz, [](double value) { return value * 10000.; });
    const auto diffCovDszDsz = fillCov2(
        h_diffCovDszDsz, reco::TrackBase::i_dsz, reco::TrackBase::i_dsz, [](double value) { return value * 10000.; });

    if (isInRange(diffCovDszDsz.status())) {
      fillNoFlow(h_diffDszError, diffRelative(pcRef->dzError(), track.dszError()));
      fillNoFlow(h_diffDzError, diffRelative(pcRef->dzError(), track.dzError()));
      fillNoFlow(h_diffTrackDzError, diffRelative(trackPc.dzError(), track.dzError()));
    }
    if (isInRange(diffCovDxyDxy.status())) {
      fillNoFlow(h_diffDxyError, diffRelative(pcRef->dxyError(), track.dxyError()));
      fillNoFlow(h_diffTrackDxyError, diffRelative(trackPc.dxyError(), track.dxyError()));
    }
    fillNoFlow(h_diffPtError, diffRelative(trackPc.ptError(), track.ptError()));
    fillNoFlow(h_diffEtaError, diffRelative(trackPc.etaError(), track.etaError()));

    // For the non-HitPattern ones, take into account the PackedCandidate packing precision
    const auto trackNumberOfHits = track.hitPattern().numberOfValidHits();
    const auto trackNumberOfPixelHits = track.hitPattern().numberOfValidPixelHits();
    const auto trackNumberOfStripHits = track.hitPattern().numberOfValidStripHits();
    const auto pcNumberOfHits = pcRef->numberOfHits();
    const auto pcNumberOfPixelHits = pcRef->numberOfPixelHits();
    const auto pcNumberOfStripHits = pcNumberOfHits - pcNumberOfPixelHits;
    const auto trackNumberOfLayers = track.hitPattern().trackerLayersWithMeasurement();
    const auto trackNumberOfPixelLayers = track.hitPattern().pixelLayersWithMeasurement();
    const auto trackNumberOfStripLayers = track.hitPattern().stripLayersWithMeasurement();
    const auto pcNumberOfLayers = pcRef->trackerLayersWithMeasurement();
    const auto pcNumberOfPixelLayers = pcRef->pixelLayersWithMeasurement();
    const auto pcNumberOfStripLayers = pcRef->stripLayersWithMeasurement();

    // layer number overflow (should be zero)
    const int pixelLayerOverflow = trackNumberOfPixelLayers > pat::PackedCandidate::trackPixelHitsMask
                                       ? trackNumberOfPixelLayers - pat::PackedCandidate::trackPixelHitsMask
                                       : 0;
    const int stripLayerOverflow = trackNumberOfStripLayers > pat::PackedCandidate::trackStripHitsMask
                                       ? trackNumberOfStripLayers - pat::PackedCandidate::trackStripHitsMask
                                       : 0;
    const int layerOverflow =
        trackNumberOfLayers > (pat::PackedCandidate::trackPixelHitsMask + pat::PackedCandidate::trackStripHitsMask)
            ? trackNumberOfLayers -
                  (pat::PackedCandidate::trackPixelHitsMask + pat::PackedCandidate::trackStripHitsMask)
            : 0;

    // hit overflow (should also be zero)
    const int pixelOverflow =
        trackNumberOfPixelHits - pcNumberOfPixelLayers > pat::PackedCandidate::trackPixelHitsMask
            ? trackNumberOfPixelHits - pcNumberOfPixelLayers - pat::PackedCandidate::trackPixelHitsMask
            : 0;
    const int stripOverflow =
        trackNumberOfStripHits - pcNumberOfStripLayers > pat::PackedCandidate::trackStripHitsMask
            ? trackNumberOfStripHits - pcNumberOfStripLayers - pat::PackedCandidate::trackStripHitsMask
            : 0;
    const int hitsOverflow =
        trackNumberOfHits - pcNumberOfLayers >
                (pat::PackedCandidate::trackPixelHitsMask + pat::PackedCandidate::trackStripHitsMask)
            ? trackNumberOfHits - pcNumberOfLayers -
                  (pat::PackedCandidate::trackPixelHitsMask + pat::PackedCandidate::trackStripHitsMask)
            : 0;
    // PackedCandidate counts overflow pixel hits as strip
    const int pixelInducedStripOverflow =
        (trackNumberOfStripHits + pixelOverflow - pcNumberOfStripLayers) > pat::PackedCandidate::trackStripHitsMask
            ? (trackNumberOfStripHits + pixelOverflow - stripOverflow - pcNumberOfStripLayers) -
                  pat::PackedCandidate::trackStripHitsMask
            : 0;
    h_numberPixelLayersOverMax->Fill(pixelLayerOverflow);
    h_numberStripLayersOverMax->Fill(stripLayerOverflow);
    h_numberLayersOverMax->Fill(layerOverflow);
    h_numberPixelHitsOverMax->Fill(pixelOverflow);
    h_numberStripHitsOverMax->Fill(stripOverflow);
    h_numberHitsOverMax->Fill(hitsOverflow);

    int diffNumberOfPixelHits = 0;
    int diffNumberOfHits = 0;
    int diffNumberOfPixelLayers = 0;
    int diffNumberOfStripLayers = 0;
    if (pixelLayerOverflow) {
      diffNumberOfPixelLayers = pcNumberOfPixelLayers - pat::PackedCandidate::trackPixelHitsMask;
    } else {
      diffNumberOfPixelLayers = pcNumberOfPixelLayers - trackNumberOfPixelLayers;
    }
    if (pixelOverflow) {
      diffNumberOfPixelHits = pcNumberOfPixelHits - pcNumberOfPixelLayers - pat::PackedCandidate::trackPixelHitsMask;
    } else {
      diffNumberOfPixelHits = pcNumberOfPixelHits - trackNumberOfPixelHits;
    }
    if (stripLayerOverflow) {
      diffNumberOfStripLayers = pcNumberOfStripLayers - pat::PackedCandidate::trackStripHitsMask;
    } else {
      diffNumberOfStripLayers = pcNumberOfStripLayers - trackNumberOfStripLayers;
    }
    if (stripOverflow || pixelInducedStripOverflow || pixelOverflow) {
      int diffNumberOfStripHits = 0;
      if (stripOverflow || pixelInducedStripOverflow) {
        diffNumberOfStripHits = pcNumberOfStripHits - pat::PackedCandidate::trackStripHitsMask;
      } else if (pixelOverflow) {
        diffNumberOfStripHits = (pcNumberOfStripHits - pixelOverflow) - trackNumberOfStripHits;
      }

      diffNumberOfHits = diffNumberOfPixelHits + diffNumberOfStripHits;
    } else {
      diffNumberOfHits = pcNumberOfHits - trackNumberOfHits;
    }

    fillNoFlow(h_diffNumberOfPixelHits, diffNumberOfPixelHits);
    fillNoFlow(h_diffNumberOfHits, diffNumberOfHits);
    fillNoFlow(h_diffNumberOfPixelLayers, diffNumberOfPixelLayers);
    fillNoFlow(h_diffNumberOfStripLayers, diffNumberOfStripLayers);

    int diffLostInnerHits = 0;
    const auto trackLostInnerHits = track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    switch (pcRef->lostInnerHits()) {
      case pat::PackedCandidate::validHitInFirstPixelBarrelLayer:
      case pat::PackedCandidate::noLostInnerHits:
        diffLostInnerHits = -trackLostInnerHits;
        break;
      case pat::PackedCandidate::oneLostInnerHit:
        diffLostInnerHits = 1 - trackLostInnerHits;
        break;
      case pat::PackedCandidate::moreLostInnerHits:
        diffLostInnerHits = trackLostInnerHits >= 2 ? 0 : 2 - trackLostInnerHits;
        break;
    }
    fillNoFlow(h_diffLostInnerHits, diffLostInnerHits);

    // For HitPattern ones, calculate the full diff (i.e. some differences are expected)
    auto diffHitPatternPixelLayersWithMeasurement =
        trackPc.hitPattern().pixelLayersWithMeasurement() - trackNumberOfPixelLayers;
    fillNoFlow(h_diffHitPatternPixelLayersWithMeasurement, diffHitPatternPixelLayersWithMeasurement);
    auto diffHitPatternStripLayersWithMeasurement =
        trackPc.hitPattern().stripLayersWithMeasurement() - trackNumberOfStripLayers;
    fillNoFlow(h_diffHitPatternStripLayersWithMeasurement, diffHitPatternStripLayersWithMeasurement);
    auto diffHitPatternTrackerLayersWithMeasurement =
        trackPc.hitPattern().trackerLayersWithMeasurement() - trackNumberOfLayers;
    fillNoFlow(h_diffHitPatternTrackerLayersWithMeasurement, diffHitPatternTrackerLayersWithMeasurement);
    auto diffHitPatternNumberOfValidPixelHits = trackPc.hitPattern().numberOfValidPixelHits() - trackNumberOfPixelHits;
    fillNoFlow(h_diffHitPatternNumberOfValidPixelHits, diffHitPatternNumberOfValidPixelHits);
    auto diffHitPatternNumberOfValidHits = trackPc.hitPattern().numberOfValidHits() - trackNumberOfHits;
    fillNoFlow(h_diffHitPatternNumberOfValidHits, diffHitPatternNumberOfValidHits);
    fillNoFlow(h_diffHitPatternNumberOfLostInnerHits,
               trackPc.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) -
                   track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));

    // hasValidHitInFirstPixelBarrel is set only if numberOfLostHits(MISSING_INNER_HITS) == 0
    int diffHitPatternHasValidHitInFirstPixelBarrel = 0;
    if (track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) == 0) {
      h_selectionFlow->Fill(sf_NoMissingInnerHits);
      diffHitPatternHasValidHitInFirstPixelBarrel =
          static_cast<int>(
              trackPc.hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)) -
          static_cast<int>(track.hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1));
      fillNoFlow(h_diffHitPatternHasValidHitInFirstPixelBarrel, diffHitPatternHasValidHitInFirstPixelBarrel);
    }

    // Print warning if there are differences outside the expected range
    if (debug_ &&
        (diffNormalizedChi2 < -1 || diffNormalizedChi2 > 0 || diffCharge != 0 || diffHP != 0 ||
         std::abs(diffPhi) > 5e-4 || diffDxyAssocPV.outsideExpectedRange() || diffDzAssocPV.outsideExpectedRange() ||
         std::abs(diffDxyPV) > 0.05 || std::abs(diffDzPV) > 0.05 || diffCovQoverpQoverp.outsideExpectedRange() ||
         diffCovLambdaLambda.outsideExpectedRange() || diffCovLambdaDsz.outsideExpectedRange() ||
         diffCovPhiPhi.outsideExpectedRange() || diffCovPhiDxy.outsideExpectedRange() ||
         diffCovDxyDxy.outsideExpectedRange() || diffCovDxyDsz.outsideExpectedRange() ||
         diffCovDszDsz.outsideExpectedRange() || diffNumberOfPixelHits != 0 || diffNumberOfHits != 0 ||
         diffLostInnerHits != 0 || diffHitPatternHasValidHitInFirstPixelBarrel != 0)) {
      edm::LogInfo("PackedCandidateTrackValidator")
          << "Track " << i << " pt " << track.pt() << " eta " << track.eta() << " phi " << track.phi() << " chi2 "
          << track.chi2() << " ndof " << track.ndof() << "\n"
          << "  ptError " << track.ptError() << " etaError " << track.etaError() << " phiError " << track.phiError()
          << " dxyError " << track.dxyError() << " dzError " << track.dzError() << "\n"
          << "  refpoint " << track.referencePoint() << " momentum " << track.momentum() << "\n"
          << "  dxy " << track.dxy() << " dz " << track.dz() << "\n"
          << "  " << TrackAlgoPrinter(track) << " lost inner hits " << trackLostInnerHits << " lost outer hits "
          << track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) << " hitpattern "
          << HitPatternPrinter(track) << " \n"
          << " PC " << pcRef.id() << ":" << pcRef.key() << " track pt " << trackPc.pt() << " eta " << trackPc.eta()
          << " phi " << trackPc.phi() << " (PC " << pcRef->phi() << ") chi2 " << trackPc.chi2() << " ndof "
          << trackPc.ndof() << " pdgId " << pcRef->pdgId() << " mass " << pcRef->mass() << "\n"
          << "  ptError " << trackPc.ptError() << " etaError " << trackPc.etaError() << " phiError "
          << trackPc.phiError() << "\n"
          << "  pc.vertex " << pcRef->vertex() << " momentum " << pcRef->momentum() << " track " << trackPc.momentum()
          << "\n"
          << "  dxy " << trackPc.dxy() << " dz " << trackPc.dz() << " pc.dz " << pcRef->dz() << " dxyError "
          << trackPc.dxyError() << " dzError " << trackPc.dzError() << "\n"
          << "  dxy(PV) " << trackPc.dxy(pv.position()) << " dz(PV) " << trackPc.dz(pv.position()) << " dxy(assocPV) "
          << trackPc.dxy(pcVertex.position()) << " dz(assocPV) " << trackPc.dz(pcVertex.position()) << "\n"
          << " (diff PackedCandidate track)"
          << " highPurity " << diffHP << " " << trackPc.quality(reco::TrackBase::highPurity) << " "
          << track.quality(reco::TrackBase::highPurity) << " charge " << diffCharge << " " << trackPc.charge() << " "
          << track.charge() << " normalizedChi2 " << diffNormalizedChi2 << " " << trackPc.normalizedChi2() << " "
          << track.normalizedChi2() << "\n "
          << " numberOfAllHits " << diffNumberOfHits << " " << pcNumberOfHits << " " << trackNumberOfHits
          << " numberOfPixelHits " << diffNumberOfPixelHits << " " << pcNumberOfPixelHits << " "
          << trackNumberOfPixelHits << " numberOfStripHits # " << pcNumberOfStripHits << " " << trackNumberOfStripHits
          << "\n "
          << " hitPattern.numberOfValidPixelHits " << diffHitPatternNumberOfValidPixelHits << " "
          << trackPc.hitPattern().numberOfValidPixelHits() << " " << track.hitPattern().numberOfValidPixelHits()
          << " hitPattern.numberOfValidHits " << diffHitPatternNumberOfValidHits << " "
          << trackPc.hitPattern().numberOfValidHits() << " " << track.hitPattern().numberOfValidHits()
          << " hitPattern.hasValidHitInFirstPixelBarrel " << diffHitPatternHasValidHitInFirstPixelBarrel << " "
          << trackPc.hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1) << " "
          << track.hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1) << "\n "
          << " lostInnerHits  " << diffLostInnerHits << " " << pcRef->lostInnerHits() << " #"
          << " phi (5e-4) " << diffPhi << " " << trackPc.phi() << " " << track.phi() << "\n "
          << " dxy(assocPV) " << diffDxyAssocPV << "\n "
          << " dz(assocPV) " << diffDzAssocPV << "\n "
          << " dxy(PV) (0.05) " << diffDxyPV << " " << pcRef->dxy(pv.position()) << " " << track.dxy(pv.position())
          << "\n "
          << " dz(PV) (0.05) " << diffDzPV << " " << pcRef->dz(pv.position()) << " " << track.dz(pv.position()) << "\n "
          << " cov(qoverp, qoverp)  " << diffCovQoverpQoverp << "\n "
          << " cov(lambda, lambda) " << diffCovLambdaLambda << "\n "
          << " cov(lambda, dsz) " << diffCovLambdaDsz << "\n "
          << " cov(phi, phi) " << diffCovPhiPhi << "\n "
          << " cov(phi, dxy) " << diffCovPhiDxy << "\n "
          << " cov(dxy, dxy) " << diffCovDxyDxy << "\n "
          << " cov(dxy, dsz) " << diffCovDxyDsz << "\n "
          << " cov(dsz, dsz) " << diffCovDszDsz;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PackedCandidateTrackValidator);
