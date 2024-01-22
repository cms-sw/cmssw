#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeights.h"

EcalEBPhase2TPGAmplWeights::EcalEBPhase2TPGAmplWeights()
    : w0_(0), w1_(0), w2_(0), w3_(0), w4_(0), w5_(0), w6_(0), w7_(0), w8_(0), w9_(0), w10_(0), w11_(0) {}

EcalEBPhase2TPGAmplWeights::~EcalEBPhase2TPGAmplWeights() {}

void EcalEBPhase2TPGAmplWeights::getValues(uint32_t& w0,
                                           uint32_t& w1,
                                           uint32_t& w2,
                                           uint32_t& w3,
                                           uint32_t& w4,
                                           uint32_t& w5,
                                           uint32_t& w6,
                                           uint32_t& w7,
                                           uint32_t& w8,
                                           uint32_t& w9,
                                           uint32_t& w10,
                                           uint32_t& w11) const {
  w0 = w0_;
  w1 = w1_;
  w2 = w2_;
  w3 = w3_;
  w4 = w4_;
  w5 = w5_;
  w6 = w6_;
  w7 = w7_;
  w8 = w8_;
  w9 = w9_;
  w10 = w10_;
  w11 = w11_;
}

void EcalEBPhase2TPGAmplWeights::setValues(const uint32_t& w0,
                                           const uint32_t& w1,
                                           const uint32_t& w2,
                                           const uint32_t& w3,
                                           const uint32_t& w4,
                                           const uint32_t& w5,
                                           const uint32_t& w6,
                                           const uint32_t& w7,
                                           const uint32_t& w8,
                                           const uint32_t& w9,
                                           const uint32_t& w10,
                                           const uint32_t& w11) {
  w0_ = w0;
  w1_ = w1;
  w2_ = w2;
  w3_ = w3;
  w4_ = w4;
  w5_ = w5;
  w6_ = w6;
  w7_ = w7;
  w8_ = w8;
  w9_ = w9;
  w10_ = w10;
  w11_ = w11;
}
