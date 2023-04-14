#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

EcalUncalibratedRecHit::EcalUncalibratedRecHit()
    : amplitude_(0.), amplitudeError_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), flags_(0), aux_(0) {
  const unsigned int nsample = EcalDataFrame::MAXSAMPLES;
  for (unsigned int ibx = 0; ibx < nsample; ++ibx)
    OOTamplitudes_[ibx] = 0.;
}

EcalUncalibratedRecHit::EcalUncalibratedRecHit(
    const DetId& id, float ampl, float ped, float jit, float chi2, uint32_t flags, uint32_t aux)
    : amplitude_(ampl),
      amplitudeError_(0.),
      pedestal_(ped),
      jitter_(jit),
      chi2_(chi2),
      flags_(flags),
      aux_(aux),
      id_(id) {
  const unsigned int nsample = EcalDataFrame::MAXSAMPLES;
  for (unsigned int ibx = 0; ibx < nsample; ++ibx)
    OOTamplitudes_[ibx] = 0.;
}

bool EcalUncalibratedRecHit::isSaturated() const { return EcalUncalibratedRecHit::checkFlag(kSaturated); }

float EcalUncalibratedRecHit::jitterError() const {
  // stored in ps, but return BXs to match with jitter units
  uint32_t jitterErrorBits = 0xFF & aux_;
  // all bits off --> time reco bailed out (return negative value)
  if ((0xFF & jitterErrorBits) == 0x00)
    return -1;
  // all bits on  --> time error over 5 ns (return large value)
  if ((0xFF & jitterErrorBits) == 0xFF)
    return 10000;

  float LSB = 1.26008;
  uint8_t exponent = jitterErrorBits >> 5;
  uint8_t significand = jitterErrorBits & ~(0x7 << 5);
  return (float)(std::pow(2, exponent) * significand * LSB) / (25. * 1000);
}

void EcalUncalibratedRecHit::setJitterError(float jitterErr) {
  // use 8 bits (3 exp, 5 mant) and store in ps
  // has range of 5 ps - 5000 ps
  // expect input in BX units
  // all bits off --> time reco bailed out
  if (jitterErr <= 0) {
    aux_ = (~0xFF & aux_);
    return;
  }
  // all bits on  --> time error over 5 ns
  if (25 * jitterErr >= 5) {
    aux_ = (0xFF | aux_);
    return;
  }

  float LSB = 1.26008;
  float quantityInLSB = (1000 * 25 * jitterErr) / LSB;
  int log2OfQuantity = (int)(log2(quantityInLSB));
  int exponentTmp = log2OfQuantity - 4;
  uint8_t exponent = 0;
  if (exponentTmp > 0)
    exponent = exponentTmp;
  uint8_t significand = (int)(std::lround(quantityInLSB / std::pow(2, exponent)));
  uint32_t jitterErrorBits = exponent << 5 | significand;

  if ((0xFF & jitterErrorBits) == 0xFF)
    jitterErrorBits = 0xFE;
  if ((0xFF & jitterErrorBits) == 0x00)
    jitterErrorBits = 0x01;

  aux_ = (~0xFF & aux_) | (jitterErrorBits & 0xFF);
}

bool EcalUncalibratedRecHit::isJitterValid() const {
  if (jitterError() <= 0)
    return false;
  else
    return true;
}

bool EcalUncalibratedRecHit::isJitterErrorValid() const {
  if (!isJitterValid())
    return false;
  if (jitterError() >= 10000)
    return false;

  return true;
}

uint8_t EcalUncalibratedRecHit::jitterErrorBits() const {
  uint8_t jitterErrorBits = 0xFF & aux_;
  return jitterErrorBits;
}

void EcalUncalibratedRecHit::setFlagBit(EcalUncalibratedRecHit::Flags flag) {
  if (flag == kGood) {
    //then set all bits to zero;
    flags_ = 0;
    return;
  }
  // else set the flagbit
  flags_ |= 0x1 << flag;
}

bool EcalUncalibratedRecHit::checkFlag(EcalUncalibratedRecHit::Flags flag) const {
  if (flag == kGood) {
    if (!flags_)
      return true;
    else
      return false;
  }  // if all flags are unset, then hit is good
  return flags_ & (0x1 << flag);
}

// For CC Timing reco
float EcalUncalibratedRecHit::nonCorrectedTime() const {

  float encBits = static_cast<float>( jitterErrorBits() );
  float decJitter = ecalcctiming::nonCorrectedSlope*jitter_ + ecalcctiming::encodingOffest - encBits/ecalcctiming::encodingValue;
  float nonCorrectedTime = ( encBits > 1 && encBits < 254 ) ? ecalcctiming::clockToNS*decJitter : -30.0;
  return nonCorrectedTime;
}

void EcalUncalibratedRecHit::setNonCorrectedTime(const float correctedJitter, const float nonCorrectedJitter) {

  float fDiff =  ecalcctiming::nonCorrectedSlope*correctedJitter - nonCorrectedJitter + ecalcctiming::encodingOffest ;
  int encBits = std::floor((fDiff*ecalcctiming::encodingValue) + 0.5);
  int bits = ( encBits < 1 ) ? 1 : ( encBits > 254 ) ? 254 : encBits;
  aux_ = (~0xFF & aux_) | (static_cast<uint8_t>(bits) & 0xFF);
}

