#include <cmath>

#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

namespace {
  float convert(float const x, int const i, HcalQIECoder const& coder, HcalQIEShape const& shape) {
    int const x1 = static_cast<int>(std::floor(x));
    int const x2 = static_cast<int>(std::floor(x + 1));
    float const y2 = coder.charge(shape, x2, i);
    float const y1 = coder.charge(shape, x1, i);
    return (y2 - y1) * (x - x1) + y1;
  }
}  // namespace

// FIXME: add proper getters to conditions
HcalConvertedPedestalsGPU::HcalConvertedPedestalsGPU(HcalPedestals const& pedestals,
                                                     HcalQIEData const& qieData,
                                                     HcalQIETypes const& qieTypes)
    : totalChannels_{pedestals.getAllContainers()[0].second.size() + pedestals.getAllContainers()[1].second.size()},
      offsetForHashes_{static_cast<uint32_t>(pedestals.getAllContainers()[0].second.size())},
      values_(totalChannels_ * 4) {
#ifdef HCAL_MAHI_CPUDEBUG
  std::cout << "hello from converted pedestals" << std::endl;
  std::cout << "pedestals HB values = " << pedestals.getAllContainers()[0].second.size()
            << "  HE values = " << pedestals.getAllContainers()[1].second.size() << std::endl;
  std::cout << "qiedata HB values = " << qieData.getAllContainers()[0].second.size()
            << "  HE values = " << qieData.getAllContainers()[1].second.size() << std::endl;
#endif

  // retrieve all collections
  auto const pedestalsAll = pedestals.getAllContainers();
  auto const qieDataAll = qieData.getAllContainers();
  auto const qieTypesAll = qieTypes.getAllContainers();

  // have to convert to fc if stored in adc
  auto const unitIsADC = pedestals.isADC();

  // fill in barrel
  auto const& pedestalBarrelValues = pedestalsAll[0].second;
  auto const& qieDataBarrelValues = qieDataAll[0].second;
  auto const& qieTypesBarrelValues = qieTypesAll[0].second;

#ifdef HCAL_MAHI_CPUDEBUG
  assert(pedestalBarrelValues.size() == qieDataBarrelValues.size());
  assert(pedestalBarrelValues.size() == qieTypesBarrelValues.size());
#endif

  for (uint64_t i = 0; i < pedestalBarrelValues.size(); ++i) {
    auto const& qieCoder = qieDataBarrelValues[i];
    auto const qieType = qieTypesBarrelValues[i].getValue() > 1 ? 1 : 0;
    auto const& qieShape = qieData.getShape(qieType);

    values_[i * 4] = unitIsADC ? convert(pedestalBarrelValues[i].getValue(0), 0, qieCoder, qieShape)
                               : pedestalBarrelValues[i].getValue(0);
    values_[i * 4 + 1] = unitIsADC ? convert(pedestalBarrelValues[i].getValue(1), 1, qieCoder, qieShape)
                                   : pedestalBarrelValues[i].getValue(1);
    values_[i * 4 + 2] = unitIsADC ? convert(pedestalBarrelValues[i].getValue(2), 2, qieCoder, qieShape)
                                   : pedestalBarrelValues[i].getValue(2);
    values_[i * 4 + 3] = unitIsADC ? convert(pedestalBarrelValues[i].getValue(3), 3, qieCoder, qieShape)
                                   : pedestalBarrelValues[i].getValue(3);
  }

  // fill in endcap
  auto const& pedestalEndcapValues = pedestalsAll[1].second;
  auto const& qieDataEndcapValues = qieDataAll[1].second;
  auto const& qieTypesEndcapValues = qieTypesAll[1].second;

#ifdef HCAL_MAHI_CPUDEBUG
  assert(pedestalEndcapValues.size() == qieDataEndcapValues.size());
  assert(pedestalEndcapValues.size() == qieTypesEndcapValues.size());
#endif

  auto const offset = pedestalBarrelValues.size();
  for (uint64_t i = 0; i < pedestalEndcapValues.size(); ++i) {
    auto const& qieCoder = qieDataEndcapValues[i];
    auto const qieType = qieTypesEndcapValues[i].getValue() > 1 ? 1 : 0;
    auto const& qieShape = qieData.getShape(qieType);
    auto const off = offset + i;

    values_[off * 4] = unitIsADC ? convert(pedestalEndcapValues[i].getValue(0), 0, qieCoder, qieShape)
                                 : pedestalEndcapValues[i].getValue(0);
    values_[off * 4 + 1] = unitIsADC ? convert(pedestalEndcapValues[i].getValue(1), 1, qieCoder, qieShape)
                                     : pedestalEndcapValues[i].getValue(1);
    values_[off * 4 + 2] = unitIsADC ? convert(pedestalEndcapValues[i].getValue(2), 2, qieCoder, qieShape)
                                     : pedestalEndcapValues[i].getValue(2);
    values_[off * 4 + 3] = unitIsADC ? convert(pedestalEndcapValues[i].getValue(3), 3, qieCoder, qieShape)
                                     : pedestalEndcapValues[i].getValue(3);

#ifdef HCAL_MAHI_CPUDEBUG
    if (pedestalEndcapValues[i].rawId() == DETID_TO_DEBUG) {
      printf("qietype = %d\n", qieType);
      printf("ped0 = %f ped1 = %f ped2 = %f ped3 = %f\n",
             pedestalEndcapValues[i].getValue(0),
             pedestalEndcapValues[i].getValue(1),
             pedestalEndcapValues[i].getValue(2),
             pedestalEndcapValues[i].getValue(3));
      printf("converted: ped0 = %f ped1 = %f ped2 = %f ped3 = %f\n",
             values_[off * 4],
             values_[off * 4 + 1],
             values_[off * 4 + 2],
             values_[off * 4 + 3]);
    }
#endif
  }
}

HcalConvertedPedestalsGPU::Product const& HcalConvertedPedestalsGPU::getProduct(cudaStream_t stream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      stream, [this](HcalConvertedPedestalsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.values, values_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalConvertedPedestalsGPU);
