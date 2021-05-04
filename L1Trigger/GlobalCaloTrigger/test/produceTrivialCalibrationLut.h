#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include <vector>
#include <memory>

class L1GctJetFinderParams;

class produceTrivialCalibrationLut {
public:
  //Typedefs
  typedef std::shared_ptr<L1GctJetEtCalibrationLut> lutPtr;
  typedef std::vector<lutPtr> lutPtrVector;

  produceTrivialCalibrationLut();
  ~produceTrivialCalibrationLut();

  void setPowerSeriesCorrectionType();
  void setOrcaStyleCorrectionType();

  const lutPtrVector produce() const;

  const L1GctJetFinderParams* jfPars() const { return m_jfPars; }

private:
  // PARAMETERS TO BE STORED IN THE CalibrationFunction
  /// scale and threshold parameters
  double m_htScaleLSB;

  /// the calibration function - converts jet Et to linear
  std::vector<std::vector<double> > m_jetCalibFunc;
  std::vector<std::vector<double> > m_tauCalibFunc;

  double m_jetEtScaleInputLsb;
  std::vector<double> m_jetEtThresholds;

  unsigned m_corrFunType;
  L1GctJetFinderParams* m_jfPars;

  /// member functions to set up the ORCA-style calibrations (if needed)
  /// (Copied from L1TriggerConfig/GctConfigProducers
  void setOrcaStyleParams();
  void setOrcaStyleParamsForBin(std::vector<double>& paramsForBin);

  void setupJfPars();
};
