#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DataFormats/Math/interface/liblogintpack.h"
using namespace logintpack;

CovarianceParameterization pat::PackedCandidate::covarianceParameterization_;
std::once_flag pat::PackedCandidate::covariance_load_flag;

void pat::PackedCandidate::pack(bool unpackAfterwards) {
  float unpackedPt = std::min<float>(p4_.load()->Pt(), MiniFloatConverter::max());
  packedPt_ = MiniFloatConverter::float32to16(unpackedPt);
  packedEta_ = int16_t(std::round(p4_.load()->Eta() / 6.0f * std::numeric_limits<int16_t>::max()));
  packedPhi_ = int16_t(std::round(p4_.load()->Phi() / 3.2f * std::numeric_limits<int16_t>::max()));
  packedM_ = MiniFloatConverter::float32to16(p4_.load()->M());
  if (unpackAfterwards) {
    delete p4_.exchange(nullptr);
    delete p4c_.exchange(nullptr);
    unpack();  // force the values to match with the packed ones
  }
}

void pat::PackedCandidate::packVtx(bool unpackAfterwards) {
  reco::VertexRef pvRef = vertexRef();
  Point pv = pvRef.isNonnull() ? pvRef->position() : Point();
  float dxPV = vertex_.load()->X() - pv.X(),
        dyPV = vertex_.load()->Y() - pv.Y();  //, rPV = std::hypot(dxPV, dyPV);
  float s = std::sin(float(p4_.load()->Phi()) + dphi_),
        c = std::cos(float(p4_.load()->Phi() + dphi_));  // not the fastest option, but we're in reduced
                                                         // precision already, so let's avoid more roundoffs
  dxy_ = -dxPV * s + dyPV * c;
  // if we want to go back to the full x,y,z we need to store also
  // float dl = dxPV * c + dyPV * s;
  // float xRec = - dxy_ * s + dl * c, yRec = dxy_ * c + dl * s;
  float pzpt = p4_.load()->Pz() / p4_.load()->Pt();
  dz_ = vertex_.load()->Z() - pv.Z() - (dxPV * c + dyPV * s) * pzpt;
  packedDxy_ = MiniFloatConverter::float32to16(dxy_ * 100);
  packedDz_ = pvRef.isNonnull() ? MiniFloatConverter::float32to16(dz_ * 100)
                                : int16_t(std::round(dz_ / 40.f * std::numeric_limits<int16_t>::max()));
  packedDPhi_ = int16_t(std::round(dphi_ / 3.2f * std::numeric_limits<int16_t>::max()));
  packedDEta_ = MiniFloatConverter::float32to16(deta_);
  packedDTrkPt_ = MiniFloatConverter::float32to16(dtrkpt_);

  if (unpackAfterwards) {
    delete vertex_.exchange(nullptr);
    unpackVtx();
  }
}

void pat::PackedCandidate::unpack() const {
  float pt = MiniFloatConverter::float16to32(packedPt_);
  double shift = (pt < 1. ? 0.1 * pt : 0.1 / pt);    // shift particle phi to break
                                                     // degeneracies in angular separations
  double sign = ((int(pt * 10) % 2 == 0) ? 1 : -1);  // introduce a pseudo-random sign of the shift
  double phi = int16_t(packedPhi_) * 3.2f / std::numeric_limits<int16_t>::max() +
               sign * shift * 3.2 / std::numeric_limits<int16_t>::max();
  auto p4 = std::make_unique<PolarLorentzVector>(pt,
                                                 int16_t(packedEta_) * 6.0f / std::numeric_limits<int16_t>::max(),
                                                 phi,
                                                 MiniFloatConverter::float16to32(packedM_));
  auto p4c = std::make_unique<LorentzVector>(*p4);
  PolarLorentzVector *expectp4 = nullptr;
  if (p4_.compare_exchange_strong(expectp4, p4.get())) {
    p4.release();
  }

  // p4c_ works as the guard for unpacking so it
  // must be set last
  LorentzVector *expectp4c = nullptr;
  if (p4c_.compare_exchange_strong(expectp4c, p4c.get())) {
    p4c.release();
  }
}

void pat::PackedCandidate::packCovariance(const reco::TrackBase::CovarianceMatrix &m, bool unpackAfterwards) {
  packedCovariance_.dptdpt = packCovarianceElement(m, 0, 0);
  packedCovariance_.detadeta = packCovarianceElement(m, 1, 1);
  packedCovariance_.dphidphi = packCovarianceElement(m, 2, 2);
  packedCovariance_.dxydxy = packCovarianceElement(m, 3, 3);
  packedCovariance_.dzdz = packCovarianceElement(m, 4, 4);
  packedCovariance_.dxydz = packCovarianceElement(m, 3, 4);
  packedCovariance_.dlambdadz = packCovarianceElement(m, 1, 4);
  packedCovariance_.dphidxy = packCovarianceElement(m, 2, 3);
  // unpack afterwards
  if (unpackAfterwards)
    unpackCovariance();
}

void pat::PackedCandidate::unpackCovariance() const {
  const CovarianceParameterization &p = covarianceParameterization();
  if (p.isValid()) {
    auto m = std::make_unique<reco::TrackBase::CovarianceMatrix>();
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++) {
        (*m)(i, j) = 0;
      }
    unpackCovarianceElement(*m, packedCovariance_.dptdpt, 0, 0);
    unpackCovarianceElement(*m, packedCovariance_.detadeta, 1, 1);
    unpackCovarianceElement(*m, packedCovariance_.dphidphi, 2, 2);
    unpackCovarianceElement(*m, packedCovariance_.dxydxy, 3, 3);
    unpackCovarianceElement(*m, packedCovariance_.dzdz, 4, 4);
    unpackCovarianceElement(*m, packedCovariance_.dxydz, 3, 4);
    unpackCovarianceElement(*m, packedCovariance_.dlambdadz, 1, 4);
    unpackCovarianceElement(*m, packedCovariance_.dphidxy, 2, 3);
    reco::TrackBase::CovarianceMatrix *expected = nullptr;
    if (m_.compare_exchange_strong(expected, m.get())) {
      m.release();
    }

  } else {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "You do not have a valid track parameters file loaded. "
        << "Please check that the release version is compatible with your "
           "input data"
        << "or avoid accessing track parameter uncertainties. ";
  }
}

void pat::PackedCandidate::unpackVtx() const {
  reco::VertexRef pvRef = vertexRef();
  dphi_ = int16_t(packedDPhi_) * 3.2f / std::numeric_limits<int16_t>::max(),
  deta_ = MiniFloatConverter::float16to32(packedDEta_);
  dtrkpt_ = MiniFloatConverter::float16to32(packedDTrkPt_);
  dxy_ = MiniFloatConverter::float16to32(packedDxy_) / 100.;
  dz_ = pvRef.isNonnull() ? MiniFloatConverter::float16to32(packedDz_) / 100.
                          : int16_t(packedDz_) * 40.f / std::numeric_limits<int16_t>::max();
  Point pv = pvRef.isNonnull() ? pvRef->position() : Point();
  float phi = p4_.load()->Phi() + dphi_, s = std::sin(phi), c = std::cos(phi);
  auto vertex = std::make_unique<Point>(pv.X() - dxy_ * s,
                                        pv.Y() + dxy_ * c,
                                        pv.Z() + dz_);  // for our choice of using the PCA to the PV, by definition the
                                                        // remaining term -(dx*cos(phi) + dy*sin(phi))*(pz/pt) is zero

  Point *expected = nullptr;
  if (vertex_.compare_exchange_strong(expected, vertex.get())) {
    vertex.release();
  }
}

pat::PackedCandidate::~PackedCandidate() {
  delete p4_.load();
  delete p4c_.load();
  delete vertex_.load();
  delete track_.load();
  delete m_.load();
}

float pat::PackedCandidate::dxy(const Point &p) const {
  maybeUnpackBoth();
  const float phi = float(p4_.load()->Phi()) + dphi_;
  return -(vertex_.load()->X() - p.X()) * std::sin(phi) + (vertex_.load()->Y() - p.Y()) * std::cos(phi);
}
float pat::PackedCandidate::dz(const Point &p) const {
  maybeUnpackBoth();
  const float phi = float(p4_.load()->Phi()) + dphi_;
  const float pzpt = deta_ ? std::sinh(etaAtVtx()) : p4_.load()->Pz() / p4_.load()->Pt();
  return (vertex_.load()->Z() - p.Z()) -
         ((vertex_.load()->X() - p.X()) * std::cos(phi) + (vertex_.load()->Y() - p.Y()) * std::sin(phi)) * pzpt;
}

void pat::PackedCandidate::unpackTrk() const {
  maybeUnpackBoth();
  math::RhoEtaPhiVector p3(ptTrk(), etaAtVtx(), phiAtVtx());
  maybeUnpackCovariance();
  int numberOfStripLayers = stripLayersWithMeasurement(), numberOfPixelLayers = pixelLayersWithMeasurement();
  int numberOfPixelHits = this->numberOfPixelHits();
  int numberOfHits = this->numberOfHits();

  int ndof = numberOfHits + numberOfPixelHits - 5;
  LostInnerHits innerLost = lostInnerHits();

  auto track = std::make_unique<reco::Track>(normalizedChi2_ * ndof,
                                             ndof,
                                             *vertex_,
                                             math::XYZVector(p3.x(), p3.y(), p3.z()),
                                             charge(),
                                             *(m_.load()),
                                             reco::TrackBase::undefAlgorithm,
                                             reco::TrackBase::loose);
  int i = 0;
  if (firstHit_ == 0) {  // Backward compatible
    if (innerLost == validHitInFirstPixelBarrelLayer) {
      track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, 1, 0, TrackingRecHit::valid);
      i = 1;
    }
  } else {
    track->appendHitPattern(firstHit_, TrackingRecHit::valid);
  }

  if (firstHit_ != 0 && reco::HitPattern::pixelHitFilter(firstHit_))
    i = 1;

  // add hits to match the number of laters and validHitInFirstPixelBarrelLayer
  if (innerLost == validHitInFirstPixelBarrelLayer) {
    // then to encode the number of layers, we add more hits on distinct layers
    // (B2, B3, B4, F1, ...)
    for (; i < numberOfPixelLayers; i++) {
      if (i <= 3) {
        track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, i + 1, 0, TrackingRecHit::valid);
      } else {
        track->appendTrackerHitPattern(PixelSubdetector::PixelEndcap, i - 3, 0, TrackingRecHit::valid);
      }
    }
  } else {
    // to encode the information on the layers, we add one valid hits per layer
    // but skipping PXB1
    int iOffset = 0;
    if (firstHit_ != 0 && reco::HitPattern::pixelHitFilter(firstHit_)) {
      iOffset = reco::HitPattern::getLayer(firstHit_);
      if (reco::HitPattern::getSubStructure(firstHit_) == PixelSubdetector::PixelEndcap)
        iOffset += 3;
    } else {
      iOffset = 1;
    }
    for (; i < numberOfPixelLayers; i++) {
      if (i + iOffset <= 2) {
        track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, i + iOffset + 1, 0, TrackingRecHit::valid);
      } else {
        track->appendTrackerHitPattern(PixelSubdetector::PixelEndcap, i + iOffset - 3 + 1, 0, TrackingRecHit::valid);
      }
    }
  }
  // add extra hits (overlaps, etc), all on the first layer with a hit - to
  // avoid increasing the layer count
  for (; i < numberOfPixelHits; i++) {
    if (firstHit_ != 0 && reco::HitPattern::pixelHitFilter(firstHit_)) {
      track->appendTrackerHitPattern(reco::HitPattern::getSubStructure(firstHit_),
                                     reco::HitPattern::getLayer(firstHit_),
                                     0,
                                     TrackingRecHit::valid);
    } else {
      track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel,
                                     (innerLost == validHitInFirstPixelBarrelLayer ? 1 : 2),
                                     0,
                                     TrackingRecHit::valid);
    }
  }
  // now start adding strip layers, putting one hit on each layer so that the
  // hitPattern.stripLayersWithMeasurement works. we don't know what the layers
  // where, so we just start with TIB (4 layers), then TOB (6 layers), then TEC
  // (9) and then TID(3), so that we can get a number of valid strip layers up
  // to 4+6+9+3
  if (firstHit_ != 0 && reco::HitPattern::stripHitFilter(firstHit_))
    i += 1;
  int slOffset = 0;
  if (firstHit_ != 0 && reco::HitPattern::stripHitFilter(firstHit_)) {
    slOffset = reco::HitPattern::getLayer(firstHit_) - 1;
    if (reco::HitPattern::getSubStructure(firstHit_) == StripSubdetector::TID)
      slOffset += 4;
    if (reco::HitPattern::getSubStructure(firstHit_) == StripSubdetector::TOB)
      slOffset += 7;
    if (reco::HitPattern::getSubStructure(firstHit_) == StripSubdetector::TEC)
      slOffset += 13;
  }
  for (int sl = slOffset; sl < numberOfStripLayers + slOffset; ++sl, ++i) {
    if (sl < 4)
      track->appendTrackerHitPattern(StripSubdetector::TIB, sl + 1, 1, TrackingRecHit::valid);
    else if (sl < 4 + 3)
      track->appendTrackerHitPattern(StripSubdetector::TID, (sl - 4) + 1, 1, TrackingRecHit::valid);
    else if (sl < 7 + 6)
      track->appendTrackerHitPattern(StripSubdetector::TOB, (sl - 7) + 1, 1, TrackingRecHit::valid);
    else if (sl < 13 + 9)
      track->appendTrackerHitPattern(StripSubdetector::TEC, (sl - 13) + 1, 1, TrackingRecHit::valid);
    else
      break;  // wtf?
  }
  // finally we account for extra strip hits beyond the one-per-layer added
  // above. we put them all on TIB1, to avoid incrementing the number of
  // layersWithMeasurement.
  for (; i < numberOfHits; i++) {
    if (reco::HitPattern::stripHitFilter(firstHit_)) {
      track->appendTrackerHitPattern(reco::HitPattern::getSubStructure(firstHit_),
                                     reco::HitPattern::getLayer(firstHit_),
                                     1,
                                     TrackingRecHit::valid);
    } else {
      track->appendTrackerHitPattern(StripSubdetector::TIB, 1, 1, TrackingRecHit::valid);
    }
  }

  switch (innerLost) {
    case validHitInFirstPixelBarrelLayer:
      break;
    case noLostInnerHits:
      break;
    case oneLostInnerHit:
      track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, 1, 0, TrackingRecHit::missing_inner);
      break;
    case moreLostInnerHits:
      track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, 1, 0, TrackingRecHit::missing_inner);
      track->appendTrackerHitPattern(PixelSubdetector::PixelBarrel, 2, 0, TrackingRecHit::missing_inner);
      break;
  };

  if (trackHighPurity())
    track->setQuality(reco::TrackBase::highPurity);

  reco::Track *expected = nullptr;
  if (track_.compare_exchange_strong(expected, track.get())) {
    track.release();
  }
}

//// Everything below is just trivial implementations of reco::Candidate methods

const reco::CandidateBaseRef &pat::PackedCandidate::masterClone() const {
  throw cms::Exception("Invalid Reference") << "this Candidate has no master clone reference."
                                            << "Can't call masterClone() method.\n";
}

bool pat::PackedCandidate::hasMasterClone() const { return false; }

bool pat::PackedCandidate::hasMasterClonePtr() const { return false; }

const reco::CandidatePtr &pat::PackedCandidate::masterClonePtr() const {
  throw cms::Exception("Invalid Reference") << "this Candidate has no master clone ptr."
                                            << "Can't call masterClonePtr() method.\n";
}

size_t pat::PackedCandidate::numberOfDaughters() const { return 0; }

size_t pat::PackedCandidate::numberOfMothers() const { return 0; }

bool pat::PackedCandidate::overlap(const reco::Candidate &o) const {
  return p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
  //  return  p4() == o.p4() && charge() == o.charge();
}

const reco::Candidate *pat::PackedCandidate::daughter(size_type) const { return nullptr; }

const reco::Candidate *pat::PackedCandidate::mother(size_type) const { return nullptr; }

const reco::Candidate *pat::PackedCandidate::daughter(const std::string &) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
      << "This Candidate type does not implement daughter(std::string). "
      << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

reco::Candidate *pat::PackedCandidate::daughter(const std::string &) {
  throw edm::Exception(edm::errors::UnimplementedFeature)
      << "This Candidate type does not implement daughter(std::string). "
      << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

reco::Candidate *pat::PackedCandidate::daughter(size_type) { return nullptr; }

double pat::PackedCandidate::vertexChi2() const { return 0; }

double pat::PackedCandidate::vertexNdof() const { return 0; }

double pat::PackedCandidate::vertexNormalizedChi2() const { return 0; }

double pat::PackedCandidate::vertexCovariance(int i, int j) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
      << "reco::ConcreteCandidate does not implement vertex covariant "
         "matrix.\n";
}

void pat::PackedCandidate::fillVertexCovariance(CovarianceMatrix &err) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
      << "reco::ConcreteCandidate does not implement vertex covariant "
         "matrix.\n";
}

bool pat::PackedCandidate::longLived() const { return false; }

bool pat::PackedCandidate::massConstraint() const { return false; }

// puppiweight
void pat::PackedCandidate::setPuppiWeight(float p, float p_nolep) {
  // Set both weights at once to avoid misconfigured weights if called in the
  // wrong order
  packedPuppiweight_ = std::numeric_limits<uint8_t>::max() * p;
  packedPuppiweightNoLepDiff_ = std::numeric_limits<int8_t>::max() * (p_nolep - p);
}

float pat::PackedCandidate::puppiWeight() const {
  return 1.f * packedPuppiweight_ / std::numeric_limits<uint8_t>::max();
}

float pat::PackedCandidate::puppiWeightNoLep() const {
  return 1.f * packedPuppiweightNoLepDiff_ / std::numeric_limits<int8_t>::max() +
         1.f * packedPuppiweight_ / std::numeric_limits<uint8_t>::max();
}

void pat::PackedCandidate::setRawCaloFraction(float p) {
  if (100 * p > std::numeric_limits<uint8_t>::max())
    rawCaloFraction_ = std::numeric_limits<uint8_t>::max();  // Set to overflow value
  else
    rawCaloFraction_ = 100 * p;
}

void pat::PackedCandidate::setRawHcalFraction(float p) { rawHcalFraction_ = 100 * p; }

void pat::PackedCandidate::setCaloFraction(float p) { caloFraction_ = 100 * p; }

void pat::PackedCandidate::setHcalFraction(float p) { hcalFraction_ = 100 * p; }

void pat::PackedCandidate::setIsIsolatedChargedHadron(bool p) { isIsolatedChargedHadron_ = p; }

void pat::PackedCandidate::setDTimeAssociatedPV(float aTime, float aTimeError) {
  if (aTime == 0 && aTimeError == 0) {
    packedTime_ = 0;
    packedTimeError_ = 0;
  } else if (aTimeError == 0) {
    packedTimeError_ = 0;
    packedTime_ = packTimeNoError(aTime);
  } else {
    packedTimeError_ = packTimeError(aTimeError);
    aTimeError = unpackTimeError(packedTimeError_);  // for reproducibility
    packedTime_ = packTimeWithError(aTime, aTimeError);
  }
}

/// static to allow unit testing
uint8_t pat::PackedCandidate::packTimeError(float timeError) {
  if (timeError <= 0)
    return 0;
  // log-scale packing.
  // for MIN_TIMEERROR = 0.002, EXPO_TIMEERROR = 5:
  //      minimum value 0.002 = 2ps (packed as 1)
  //      maximum value 0.5 ns      (packed as 255)
  //      constant *relative* precision of about 2%
  return std::max<uint8_t>(
      std::min(std::round(std::ldexp(std::log2(timeError / MIN_TIMEERROR), +EXPO_TIMEERROR)), 255.f), 1);
}
float pat::PackedCandidate::unpackTimeError(uint8_t timeError) {
  return timeError > 0 ? MIN_TIMEERROR * std::exp2(std::ldexp(float(timeError), -EXPO_TIMEERROR)) : -1.0f;
}
float pat::PackedCandidate::unpackTimeNoError(int16_t time) {
  if (time == 0)
    return 0.f;
  return (time > 0 ? MIN_TIME_NOERROR : -MIN_TIME_NOERROR) *
         std::exp2(std::ldexp(float(std::abs(time)), -EXPO_TIME_NOERROR));
}
int16_t pat::PackedCandidate::packTimeNoError(float time) {
  // encoding in log scale to store times in a large range with few bits.
  // for MIN_TIME_NOERROR = 0.0002 and EXPO_TIME_NOERROR = 6:
  //    smallest non-zero time = 0.2 ps (encoded as +/-1)
  //    one BX, +/- 12.5 ns, is fully covered with 11 bits (+/- 1023)
  //    12 bits cover by far any plausible value (+/-2047 corresponds to about
  //    +/- 0.8 ms!) constant *relative* ~1% precision
  if (std::abs(time) < MIN_TIME_NOERROR)
    return 0;  // prevent underflows
  float fpacked = std::ldexp(std::log2(std::abs(time / MIN_TIME_NOERROR)), +EXPO_TIME_NOERROR);
  return (time > 0 ? +1 : -1) * std::min(std::round(fpacked), 2047.f);
}
float pat::PackedCandidate::unpackTimeWithError(int16_t time, uint8_t timeError) {
  if (time % 2 == 0) {
    // no overflow: drop rightmost bit and unpack in units of timeError
    return std::ldexp(unpackTimeError(timeError), EXPO_TIME_WITHERROR) * float(time / 2);
  } else {
    // overflow: drop rightmost bit, unpack using the noError encoding
    return pat::PackedCandidate::unpackTimeNoError(time / 2);
  }
}
int16_t pat::PackedCandidate::packTimeWithError(float time, float timeError) {
  // Encode in units of timeError * 2^EXPO_TIME_WITHERROR (~1.6% if
  // EXPO_TIME_WITHERROR = -6) the largest value that can be stored in 14 bits +
  // sign bit + overflow bit is about 260 sigmas values larger than that will be
  // stored using the no-timeError packing (with less precision). overflows of
  // these kinds should happen only for particles that are late arriving,
  // out-of-time, or mis-reconstructed, as timeError is O(20ps) and the beam
  // spot witdth is O(200ps)
  float fpacked = std::round(time / std::ldexp(timeError, EXPO_TIME_WITHERROR));
  if (std::abs(fpacked) < 16383.f) {  // 16383 = (2^14 - 1) = largest absolute
                                      // value for a signed 15 bit integer
    return int16_t(fpacked) * 2;      // make it even, and fit in a signed 16 bit int
  } else {
    int16_t packed = packTimeNoError(time);    // encode
    return packed * 2 + (time > 0 ? +1 : -1);  // make it odd, to signal that there was an overlow
  }
}
