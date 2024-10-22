#include "L1Trigger/L1THGCal/interface/HGCalVFECompressionImpl.h"

#include "FWCore/Utilities/interface/Exception.h"

HGCalVFECompressionImpl::HGCalVFECompressionImpl(const edm::ParameterSet& conf)
    : exponentBits_(conf.getParameter<uint32_t>("exponentBits")),
      mantissaBits_(conf.getParameter<uint32_t>("mantissaBits")),
      truncationBits_(conf.getParameter<uint32_t>("truncationBits")),
      rounding_(conf.getParameter<bool>("rounding")) {
  if (((1 << exponentBits_) + mantissaBits_ - 1) >= 64) {
    throw cms::Exception("CodespaceCannotFit") << "The code space cannot fit into the unsigned 64-bit space.\n";
  }
  saturationCode_ = (1 << (exponentBits_ + mantissaBits_)) - 1;
  saturationValue_ = (exponentBits_ == 0)
                         ? saturationCode_
                         : ((1ULL << (mantissaBits_ + truncationBits_ + 1)) - 1) << ((1 << exponentBits_) - 2);
}

void HGCalVFECompressionImpl::compressSingle(const uint64_t value,
                                             uint32_t& compressedCode,
                                             uint64_t& compressedValue) const {
  // check for saturation
  if (value > saturationValue_) {
    compressedCode = saturationCode_;
    compressedValue = saturationValue_;
    return;
  }

  // count bit length
  uint32_t bitlen;
  uint64_t shifted_value = value >> truncationBits_;
  uint64_t valcopy = shifted_value;
  for (bitlen = 0; valcopy != 0; valcopy >>= 1, bitlen++) {
  }
  if (bitlen <= mantissaBits_) {
    compressedCode = shifted_value;
    compressedValue = shifted_value << truncationBits_;
    return;
  }

  // build exponent and mantissa
  const uint32_t exponent = bitlen - mantissaBits_;
  const uint64_t mantissa = (shifted_value >> (exponent - 1)) & ~(1ULL << mantissaBits_);

  // assemble floating-point
  const uint32_t floatval = (exponent << mantissaBits_) | mantissa;

  // we will never want to round up maximum code here
  if (!rounding_ || floatval == saturationCode_) {
    compressedCode = floatval;
    compressedValue = ((1ULL << mantissaBits_) | mantissa) << (exponent - 1);
  } else {
    const bool roundup = ((shifted_value >> (exponent - 2)) & 1ULL) == 1ULL;
    if (!roundup) {
      compressedCode = floatval;
      compressedValue = ((1ULL << mantissaBits_) | mantissa) << (exponent - 1);
    } else {
      compressedCode = floatval + 1;
      uint64_t rmantissa = mantissa + 1;
      uint32_t rexponent = exponent;
      if (rmantissa >= (1ULL << mantissaBits_)) {
        rexponent++;
        rmantissa &= ~(1ULL << mantissaBits_);
      }
      compressedValue = ((1ULL << mantissaBits_) | rmantissa) << (rexponent - 1);
    }
  }
  compressedValue <<= truncationBits_;
}

void HGCalVFECompressionImpl::compress(const std::unordered_map<uint32_t, uint32_t>& payload,
                                       std::unordered_map<uint32_t, std::array<uint64_t, 2> >& compressed_payload) {
  for (const auto& item : payload) {
    const uint64_t value = static_cast<uint64_t>(item.second);
    uint32_t code(0);
    uint64_t compressed_value(0);
    compressSingle(value, code, compressed_value);
    std::array<uint64_t, 2> compressed_item = {{static_cast<uint64_t>(code), compressed_value}};
    compressed_payload.emplace(item.first, compressed_item);
  }
}
