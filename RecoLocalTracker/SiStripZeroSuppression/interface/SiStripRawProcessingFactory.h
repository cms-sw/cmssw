#ifndef RecoLocalTracker_SiStripRawProcessingFactory_h
#define RecoLocalTracker_SiStripRawProcessingFactory_h

namespace edm {class ParameterSet;}
class SiStripRawProcessingAlgorithms;
class SiStripFedZeroSuppression;
class SiStripPedestalsSubtractor;
class SiStripCommonModeNoiseSubtractor;
class SiStripAPVRestorer;
#include <memory>

class SiStripRawProcessingFactory {


 public:
  
  static std::auto_ptr<SiStripRawProcessingAlgorithms> create(const edm::ParameterSet&);

  static std::auto_ptr<SiStripFedZeroSuppression> create_Suppressor(const edm::ParameterSet&);
  static std::auto_ptr<SiStripPedestalsSubtractor> create_SubtractorPed(const edm::ParameterSet&);
  static std::auto_ptr<SiStripCommonModeNoiseSubtractor> create_SubtractorCMN(const edm::ParameterSet&);
  static std::auto_ptr<SiStripAPVRestorer> create_Restorer( const edm::ParameterSet&);
  
  static bool create_doAPVRestorer(const edm::ParameterSet&);
  static bool create_useCMMeanMap(const edm::ParameterSet&);
};
#endif
