#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFECompressionImpl.h"

HGCalVFECompressionImpl::
HGCalVFECompressionImpl(const edm::ParameterSet& conf):
    exponentBits_(conf.getParameter<uint32_t>("exponentBits")),
    mantissaBits_(conf.getParameter<uint32_t>("mantissaBits")),
    rounding_(conf.getParameter<bool>("rounding"))
{
  saturationCode_ = (1 << (exponentBits_ + mantissaBits_)) - 1;
  saturationValue_ = 0xffffffff;
  if (((1 << exponentBits_) + mantissaBits_ - 1) < 32) {
    saturationValue_ = (1 << ((1 << exponentBits_) + mantissaBits_ - 1)) - 1;
  }
  else {
    throw cms::Exception("CodespaceCannotFit")
      << "The code space cannot fit into the unsigned 32-bit space.\n";
  }
  // TODO: for non-saturating need to get maximum code point as well?
  // i.e. 8-bit compressed code that would hit 0xffffffff
}

uint32_t
HGCalVFECompressionImpl::
bitLength(uint32_t x)
{
  uint32_t bitlen;
  for (bitlen = 0; x != 0; x >>= 1, bitlen++) {}
  return bitlen;
}

uint32_t
HGCalVFECompressionImpl::
compressSingle(const uint32_t value)
{
  // check for saturation
  if (value > saturationValue_) {
    return saturationCode_;
  }

  // count bit length
  const uint32_t bitlen = bitLength(value);
  if (bitlen <= mantissaBits_) {
    return value;
  }

  // build exponent and mantissa
  const uint32_t exponent = bitlen - mantissaBits_;
  const uint32_t mantissa = (value >> (exponent-1)) & ~(1<<mantissaBits_);

  // assemble floating-point
  const uint32_t floatval = (exponent << mantissaBits_) | mantissa;

  // we will never want to round up maximum code here
  if (!rounding_ || floatval == saturationCode_) {
    return floatval;
  }
  else {
    const bool roundup = ((value >> (exponent-2)) & 1) == 1;
    return roundup ? floatval+1 : floatval;
  }
}

uint32_t
HGCalVFECompressionImpl::
decompressSingle(const uint32_t code)
{
  const uint32_t exponent = (code >> mantissaBits_) & ((1 << exponentBits_) - 1);
  if (exponent == 0) {
    return code;
  }
  const uint32_t mantissa = code & ((1 << mantissaBits_) - 1);
  return ((1 << mantissaBits_) | mantissa) << (exponent - 1);
}

void
HGCalVFECompressionImpl::
compress(const std::map<HGCalDetId, uint32_t>& payload,
               std::map<HGCalDetId, std::array<uint32_t, 2> >& compressed_payload)
{
  for (const auto& item : payload) {
    const uint32_t value = item.second;
    const uint32_t code = compressSingle(value);
    const uint32_t compressed_value = decompressSingle(code);
    std::array<uint32_t, 2> compressed_item = {{ code, compressed_value }};
    compressed_payload.insert( std::make_pair(item.first, compressed_item) );
  }
}

