#ifndef RecoLocalTracker_SiStripRawProcessingFactory_h
#define RecoLocalTracker_SiStripRawProcessingFactory_h

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm
class SiStripRawProcessingAlgorithms;
class SiStripFedZeroSuppression;
class SiStripPedestalsSubtractor;
class SiStripCommonModeNoiseSubtractor;
class SiStripAPVRestorer;
#include <memory>

class SiStripRawProcessingFactory {
public:
  static std::unique_ptr<SiStripRawProcessingAlgorithms> create(const edm::ParameterSet&, edm::ConsumesCollector);

  static std::unique_ptr<SiStripFedZeroSuppression> create_Suppressor(const edm::ParameterSet&, edm::ConsumesCollector);
  static std::unique_ptr<SiStripPedestalsSubtractor> create_SubtractorPed(const edm::ParameterSet&,
                                                                          edm::ConsumesCollector);
  static std::unique_ptr<SiStripCommonModeNoiseSubtractor> create_SubtractorCMN(const edm::ParameterSet&,
                                                                                edm::ConsumesCollector);
  static std::unique_ptr<SiStripAPVRestorer> create_Restorer(const edm::ParameterSet&, edm::ConsumesCollector);
};
#endif
