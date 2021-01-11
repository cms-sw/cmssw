#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h

#include <cstdint>
#include <vector>

// The main purpose of this class is to deliver digi and cluster data
// from an EDProducer that transfers the data from GPU to host to an
// EDProducer that converts the SoA to legacy data products. The class
// is independent of any GPU technology, and in prunciple could be
// produced by host code, and be used for other purposes than
// conversion-to-legacy as well.
class SiPixelDigisSoA {
public:
  SiPixelDigisSoA() = default;
  explicit SiPixelDigisSoA(
      size_t nDigis, const uint32_t* pdigi, const uint32_t* rawIdArr, const uint16_t* adc, const int32_t* clus);
  ~SiPixelDigisSoA() = default;

  auto size() const { return pdigi_.size(); }

  uint32_t pdigi(size_t i) const { return pdigi_[i]; }
  uint32_t rawIdArr(size_t i) const { return rawIdArr_[i]; }
  uint16_t adc(size_t i) const { return adc_[i]; }
  int32_t clus(size_t i) const { return clus_[i]; }

  const std::vector<uint32_t>& pdigiVector() const { return pdigi_; }
  const std::vector<uint32_t>& rawIdArrVector() const { return rawIdArr_; }
  const std::vector<uint16_t>& adcVector() const { return adc_; }
  const std::vector<int32_t>& clusVector() const { return clus_; }

private:
  std::vector<uint32_t> pdigi_;     // packed digi (row, col, adc) of each pixel
  std::vector<uint32_t> rawIdArr_;  // DetId of each pixel
  std::vector<uint16_t> adc_;       // ADC of each pixel
  std::vector<int32_t> clus_;       // cluster id of each pixel
};

#endif
