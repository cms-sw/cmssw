#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include <vector>

class produceTrivialCalibrationLut
{
 public:
  produceTrivialCalibrationLut();
  ~produceTrivialCalibrationLut();

  void setPowerSeriesCorrectionType();
  void setOrcaStyleCorrectionType();
  void setOldOrcaStyleCorrectionType();

  L1GctJetEtCalibrationLut* produce();

 private:

  // PARAMETERS TO BE STORED IN THE CalibrationFunction
  /// scale and threshold parameters
  double m_htScaleLSB;
  double m_threshold;

  /// the calibration function - converts jet Et to linear 
  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

  double m_jetEtScaleInputLsb;
  std::vector<double> m_jetEtThresholds;

  L1GctJetEtCalibrationFunction::CorrectionFunctionType m_corrFunType; 
    
  /// member functions to set up the ORCA-style calibrations (if needed)
  /// (Copied from L1TriggerConfig/GctConfigProducers
  void setOrcaStyleParams();
  void setOrcaStyleParamsForBin(std::vector<double>& paramsForBin);

};

