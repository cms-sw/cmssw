#include "L1Trigger/L1THGCal/interface/HGCalVFECompressionImpl.h"

HGCalVFECompressionImpl::HGCalVFECompressionImpl(const edm::ParameterSet& conf)
    : exponentBits_(conf.getParameter<uint32_t>("exponentBits")),
      mantissaBits_(conf.getParameter<uint32_t>("mantissaBits")),
      rounding_(conf.getParameter<bool>("rounding")) {
  if (((1 << exponentBits_) + mantissaBits_ - 1) >= 32) {
    throw cms::Exception("CodespaceCannotFit") << "The code space cannot fit into the unsigned 32-bit space.\n";
  }
  saturationCode_ = (1 << (exponentBits_ + mantissaBits_)) - 1;
  saturationValue_ =
      (exponentBits_ == 0) ? saturationCode_ : ((1 << (mantissaBits_ + 1)) - 1) << ((1 << exponentBits_) - 2);
}

void HGCalVFECompressionImpl::compressSingle(const uint32_t value,
                                             uint32_t& compressedCode,
                                             uint32_t& compressedValue) const {
  // check for saturation
  if (value > saturationValue_) {
    compressedCode = saturationCode_;
    compressedValue = saturationValue_;
    return;
  }

  // count bit length
  uint32_t bitlen;
  uint32_t valcopy = value;
  for (bitlen = 0; valcopy != 0; valcopy >>= 1, bitlen++) {
  }
  if (bitlen <= mantissaBits_) {
    compressedCode = value;
    compressedValue = value;
    return;
  }

  // build exponent and mantissa
  const uint32_t exponent = bitlen - mantissaBits_;
  const uint32_t mantissa = (value >> (exponent - 1)) & ~(1 << mantissaBits_);

  // assemble floating-point
  const uint32_t floatval = (exponent << mantissaBits_) | mantissa;

  // we will never want to round up maximum code here
  if (!rounding_ || floatval == saturationCode_) {
    compressedCode = floatval;
    compressedValue = ((1 << mantissaBits_) | mantissa) << (exponent - 1);
  } else {
    const bool roundup = ((value >> (exponent - 2)) & 1) == 1;
    if (!roundup) {
      compressedCode = floatval;
      compressedValue = ((1 << mantissaBits_) | mantissa) << (exponent - 1);
    } else {
      compressedCode = floatval + 1;
      uint32_t rmantissa = mantissa + 1;
      uint32_t rexponent = exponent;
      if (rmantissa >= (1U << mantissaBits_)) {
        rexponent++;
        rmantissa &= ~(1 << mantissaBits_);
      }
      compressedValue = ((1 << mantissaBits_) | rmantissa) << (rexponent - 1);
    }
  }
}

void HGCalVFECompressionImpl::compress(const std::unordered_map<uint32_t, uint32_t>& payload,
                                       std::unordered_map<uint32_t, std::array<uint32_t, 2> >& compressed_payload) {
  for (const auto& item : payload) {
    const uint32_t value = item.second;
    uint32_t code(0);
    uint32_t compressed_value(0);
    compressSingle(value, code, compressed_value);
    std::array<uint32_t, 2> compressed_item = {{code, compressed_value}};
    compressed_payload.emplace(item.first, compressed_item);
  }
}
