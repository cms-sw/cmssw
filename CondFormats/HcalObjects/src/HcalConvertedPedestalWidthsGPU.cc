#include <cmath>

#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalWidthsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

namespace {
  float convert(
      float const value, float const width, int const i, HcalQIECoder const& coder, HcalQIEShape const& shape) {
    float const y = value;
    float const x = width;
    unsigned const x1 = static_cast<unsigned>(std::floor(y));
    unsigned const x2 = static_cast<unsigned>(std::floor(y + 1.));
    unsigned iun = static_cast<unsigned>(i);
    float const y1 = coder.charge(shape, x1, iun);
    float const y2 = coder.charge(shape, x2, iun);
    return (y2 - y1) * x;
  }
}  // namespace

// FIXME: add proper getters to conditions
HcalConvertedPedestalWidthsGPU::HcalConvertedPedestalWidthsGPU(HcalPedestals const& pedestals,
                                                               HcalPedestalWidths const& pedestalWidths,
                                                               HcalQIEData const& qieData,
                                                               HcalQIETypes const& qieTypes)
    : totalChannels_{pedestals.getAllContainers()[0].second.size() + pedestals.getAllContainers()[1].second.size()},
      values_(totalChannels_ * 4) {
#ifdef HCAL_MAHI_CPUDEBUG
  std::cout << "hello from converted pedestal widths" << std::endl;
  std::cout << "pedestals HB values = " << pedestals.getAllContainers()[0].second.size()
            << "  HE values = " << pedestals.getAllContainers()[1].second.size() << std::endl;
  std::cout << "qiedata HB values = " << qieData.getAllContainers()[0].second.size()
            << "  HE values = " << qieData.getAllContainers()[1].second.size() << std::endl;
#endif

  // retrieve all collections
  auto const pedestalsAll = pedestals.getAllContainers();
  auto const pedestalWidthsAll = pedestalWidths.getAllContainers();
  auto const qieDataAll = qieData.getAllContainers();
  auto const qieTypesAll = qieTypes.getAllContainers();

  // have to convert to fc if stored in adc
  auto const unitIsADC = pedestals.isADC();

  // fill in barrel
  auto const& pedestalBarrelValues = pedestalsAll[0].second;
  auto const& pedestalWidthBarrelValues = pedestalWidthsAll[0].second;
  auto const& qieDataBarrelValues = qieDataAll[0].second;
  auto const& qieTypesBarrelValues = qieTypesAll[0].second;

#ifdef HCAL_MAHI_CPUDEBUG
  assert(pedestalWidthBarrelValues.size() == pedestalBarrelValues.size());
  assert(pedestalBarrelValues.size() == qieDataBarrelValues.size());
  assert(pedestalBarrelValues.size() == qieTypesBarrelValues.size());
#endif

  for (uint64_t i = 0; i < pedestalBarrelValues.size(); ++i) {
    auto const& qieCoder = qieDataBarrelValues[i];
    auto const qieType = qieTypesBarrelValues[i].getValue() > 1 ? 1 : 0;
    auto const& qieShape = qieData.getShape(qieType);

    values_[i * 4] =
        unitIsADC
            ? convert(
                  pedestalBarrelValues[i].getValue(0), pedestalWidthBarrelValues[i].getWidth(0), 0, qieCoder, qieShape)
            : pedestalWidthBarrelValues[i].getWidth(0);
    values_[i * 4 + 1] =
        unitIsADC
            ? convert(
                  pedestalBarrelValues[i].getValue(1), pedestalWidthBarrelValues[i].getWidth(1), 1, qieCoder, qieShape)
            : pedestalWidthBarrelValues[i].getWidth(1);
    values_[i * 4 + 2] =
        unitIsADC
            ? convert(
                  pedestalBarrelValues[i].getValue(2), pedestalWidthBarrelValues[i].getWidth(2), 2, qieCoder, qieShape)
            : pedestalWidthBarrelValues[i].getWidth(2);
    values_[i * 4 + 3] =
        unitIsADC
            ? convert(
                  pedestalBarrelValues[i].getValue(3), pedestalWidthBarrelValues[i].getWidth(3), 3, qieCoder, qieShape)
            : pedestalWidthBarrelValues[i].getWidth(3);
  }

  // fill in endcap
  auto const& pedestalEndcapValues = pedestalsAll[1].second;
  auto const& pedestalWidthEndcapValues = pedestalWidthsAll[1].second;
  auto const& qieDataEndcapValues = qieDataAll[1].second;
  auto const& qieTypesEndcapValues = qieTypesAll[1].second;

#ifdef HCAL_MAHI_CPUDEBUG
  assert(pedestalWidthEndcapValues.size() == pedestalEndcapValues.size());
  assert(pedestalEndcapValues.size() == qieDataEndcapValues.size());
  assert(pedestalEndcapValues.size() == qieTypesEndcapValues.size());
#endif

  auto const offset = pedestalWidthBarrelValues.size();
  for (uint64_t i = 0; i < pedestalEndcapValues.size(); ++i) {
    auto const& qieCoder = qieDataEndcapValues[i];
    auto const qieType = qieTypesEndcapValues[i].getValue() > 1 ? 1 : 0;
    auto const& qieShape = qieData.getShape(qieType);
    auto const off = offset + i;

    values_[off * 4] =
        unitIsADC
            ? convert(
                  pedestalEndcapValues[i].getValue(0), pedestalWidthEndcapValues[i].getWidth(0), 0, qieCoder, qieShape)
            : pedestalWidthEndcapValues[i].getWidth(0);
    values_[off * 4 + 1] =
        unitIsADC
            ? convert(
                  pedestalEndcapValues[i].getValue(1), pedestalWidthEndcapValues[i].getWidth(1), 1, qieCoder, qieShape)
            : pedestalWidthEndcapValues[i].getWidth(1);
    values_[off * 4 + 2] =
        unitIsADC
            ? convert(
                  pedestalEndcapValues[i].getValue(2), pedestalWidthEndcapValues[i].getWidth(2), 2, qieCoder, qieShape)
            : pedestalWidthEndcapValues[i].getWidth(2);
    values_[off * 4 + 3] =
        unitIsADC
            ? convert(
                  pedestalEndcapValues[i].getValue(3), pedestalWidthEndcapValues[i].getWidth(3), 3, qieCoder, qieShape)
            : pedestalWidthEndcapValues[i].getWidth(3);

#ifdef HCAL_MAHI_CPUDEBUG
    if (pedestalEndcapValues[i].rawId() == DETID_TO_DEBUG) {
      for (int i = 0; i < 4; i++)
        printf("pedestalWidth(%d) = %f original pedestalWidth(%d) = %f\n",
               i,
               values_[off * 4 + i],
               i,
               pedestalWidthEndcapValues[i].getWidth(3));
    }
#endif
  }
}

HcalConvertedPedestalWidthsGPU::Product const& HcalConvertedPedestalWidthsGPU::getProduct(cudaStream_t stream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      stream, [this](HcalConvertedPedestalWidthsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.values, values_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalConvertedPedestalWidthsGPU);
