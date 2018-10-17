#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFECompressionImpl.h"

HGCalVFECompressionImpl::
HGCalVFECompressionImpl(const edm::ParameterSet& conf):
    exponentBits_(conf.getParameter<uint32_t>("exponentBits")),
    mantissaBits_(conf.getParameter<uint32_t>("mantissaBits")),
    rounding_(conf.getParameter<bool>("rounding"))
{
  saturable_ = false;
  saturationValue_ = 0xffffffff;
  if (((1 << exponentBits_) + mantissaBits_ - 1) < 32) {
    saturable_ = true;
    saturationValue_ = (1 << ((1 << exponentBits_) + mantissaBits_ - 1)) - 1;
  }
  // TODO: for non-saturating need to get maximum code point as well?
  // i.e. 8-bit compressed code that would hit 0xffffffff

  // build the compressed values lookup table
  uint32_t compval = 0;
  const uint32_t mantissaMask = (1 << mantissaBits_) - 1;
  uint32_t incval = 1;
  for (uint32_t code = 0; code < 0x100; code++) {
    compressedValueLUT_[code] = compval;
    if ((code & mantissaMask) == 0) {
      incval <<= 1;
      if (((code & ~mantissaMask) >> mantissaBits_) <= 1) {
        incval >>= 1;
      }
    }
    compval += incval;
  }
}

uint32_t
HGCalVFECompressionImpl::
compressSingle(const uint32_t value)
{
  // check for saturation
  if (saturable_ && value > saturationValue_)
      return 0xff;

  // count bit length
  uint32_t valcopy = value;
  uint32_t bitlen;
  for (bitlen = 0; valcopy != 0; valcopy >>= 1, bitlen++) {}
  if (bitlen <= mantissaBits_)
    return value;

  // build exponent and mantissa
  const uint32_t exponent = bitlen - mantissaBits_;
  const uint32_t mantissa = (value >> (exponent-1)) & ~(1<<mantissaBits_);

  // assemble floating-point
  const uint32_t floatval = (exponent << mantissaBits_) | mantissa;

  // we will never want to round up 0xff here
  if (!rounding_ || floatval == 0xff) {
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
  return compressedValueLUT_[code];
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
    std::array<uint32_t, 2> compressed_item = {{ static_cast<uint32_t>(code), compressed_value }};
    compressed_payload.insert( std::make_pair(item.first, compressed_item) );
  }
}

