#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

produceTrivialCalibrationLut::produceTrivialCalibrationLut() :
  m_htScaleLSB(1.0),
  m_threshold(5.0),
  m_jetCalibFunc(L1GctJetEtCalibrationFunction::NUMBER_ETA_VALUES),
  m_tauCalibFunc(L1GctJetEtCalibrationFunction::N_CENTRAL_ETA_VALUES),
  m_jetEtScaleInputLsb(0.5)
{
  const double jetThresholds[64]={
	0.,	10.,	12.,	14.,	15.,	18.,	20.,	22.,	24.,	25.,
	28.,	30.,	32.,	35.,	37.,	40.,	45.,	50.,	55.,	60.,	
	65.,	70.,	75.,	80.,	85.,	90.,	100.,	110., 	120.,	125.,	
	130.,	140.,	150.,	160.,	170.,	175.,	180.,	190.,	200.,	215.,	
	225.,	235.,	250.,	275.,	300.,	325.,	350.,	375.,	400.,	425.,	
	450.,	475.,	500.,	525.,	550.,	575.,	600.,	625.,	650.,	675.,	
	700.,	725.,	750.,	775.
    };
  for (unsigned i=0; i<64; i++) {
    m_jetEtThresholds.push_back(jetThresholds[i]);
  }
}

produceTrivialCalibrationLut::~produceTrivialCalibrationLut()
{
}

L1GctJetEtCalibrationLut* produceTrivialCalibrationLut::produce()
{
  L1CaloEtScale* jetScale = new L1CaloEtScale(m_jetEtScaleInputLsb, m_jetEtThresholds);
  L1GctJetEtCalibrationFunction* calibFun = new L1GctJetEtCalibrationFunction();

  calibFun->setOutputEtScale(*jetScale);
  calibFun->setParams(m_htScaleLSB, m_threshold,
                      m_jetCalibFunc,
                      m_tauCalibFunc);

  L1GctJetEtCalibrationLut* lut = L1GctJetEtCalibrationLut::setupLut(calibFun);

  return lut;
}
