#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include <math.h>
#include <cassert>

produceTrivialCalibrationLut::produceTrivialCalibrationLut() :
  m_htScaleLSB(0.5),
  m_jetCalibFunc(L1GctJetFinderParams::NUMBER_ETA_VALUES),
  m_tauCalibFunc(L1GctJetFinderParams::N_CENTRAL_ETA_VALUES),
  m_jetEtScaleInputLsb(0.5),
  m_corrFunType(1),
  m_jfPars(new L1GctJetFinderParams())
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

  setupJfPars();

}

produceTrivialCalibrationLut::~produceTrivialCalibrationLut()
{
}

void produceTrivialCalibrationLut::setPowerSeriesCorrectionType()
{
  m_corrFunType = 1;
  for (unsigned i=0; i<m_jetCalibFunc.size(); i++) {
    m_jetCalibFunc.at(i).clear();
  }
  for (unsigned i=0; i<m_tauCalibFunc.size(); i++) {
    m_tauCalibFunc.at(i).clear();
  }
  setupJfPars();
}

void produceTrivialCalibrationLut::setOrcaStyleCorrectionType()
{
  const double y[11]={
                       47.4, 49.4, 47.1, 49.3, 48.2, 42.0, 33.8, 17.1, 13.3, 12.4,  9.3
                     };
  const double A[11]={
                         -20.7,   -22.5,   -22.2,   -22.9,   -24.5,   -23.9,
                                  -22.1,    -6.6,    -4.5,    -3.8,     1.3
                     };
  const double B[11]={
                        0.7922,  0.7867,  0.7645,  0.7331,  0.7706,  0.7945,
                                 0.8202,  0.6958,  0.7071,  0.6558,  0.2719
                     };
  const double C[11]={
                       9.53E-5, 9.60E-5,12.09E-5,12.21E-5,12.80E-5,14.58E-5,
                               14.03E-5, 6.88E-5, 7.26E-5,48.90E-5,341.8E-5
                     };
  
  m_corrFunType = 2;
  for (unsigned i=0; i<m_jetCalibFunc.size(); i++) {
    m_jetCalibFunc.at(i).clear();
    m_jetCalibFunc.at(i).push_back(y[i]);
    m_jetCalibFunc.at(i).push_back(A[i]);
    m_jetCalibFunc.at(i).push_back(B[i]);
    m_jetCalibFunc.at(i).push_back(C[i]);
  }
  for (unsigned i=0; i<m_tauCalibFunc.size(); i++) {
    m_tauCalibFunc.at(i).clear();
    m_tauCalibFunc.at(i).push_back(y[i]);
    m_tauCalibFunc.at(i).push_back(A[i]);
    m_tauCalibFunc.at(i).push_back(B[i]);
    m_tauCalibFunc.at(i).push_back(C[i]);
  }
  setOrcaStyleParams();
  setupJfPars();
}

const produceTrivialCalibrationLut::lutPtrVector produceTrivialCalibrationLut::produce() const
{
  L1CaloEtScale* jetScale = new L1CaloEtScale(m_jetEtScaleInputLsb, m_jetEtThresholds);

  lutPtrVector lutVector;
  lutPtr nextLut( new L1GctJetEtCalibrationLut() );

  for (unsigned ieta=0; ieta<L1GctJetFinderParams::NUMBER_ETA_VALUES; ieta++) {
    nextLut->setEtaBin(ieta);
    nextLut->setFunction(m_jfPars);
    nextLut->setOutputEtScale(jetScale);
    lutVector.push_back(nextLut);
    nextLut.reset ( new L1GctJetEtCalibrationLut() );
  }

  return lutVector;
}

void produceTrivialCalibrationLut::setupJfPars() {

  m_jfPars->setRegionEtLsb(m_jetEtScaleInputLsb);

  m_jfPars->setJetEtCalibrationParams(m_corrFunType,
				      m_jetCalibFunc,
				      m_tauCalibFunc);

  m_jfPars->setHtSumParams(m_htScaleLSB, 0.0, 0.0);

}

//--------------------------------------------------------------------------
//
// For ORCA-style calibration, we extend the calibration function downwards
// in energy in an automated way here.
void produceTrivialCalibrationLut::setOrcaStyleParams()
{
  for (unsigned i=0; i<m_jetCalibFunc.size(); ++i) {
    setOrcaStyleParamsForBin(m_jetCalibFunc.at(i));
  }
  for (unsigned i=0; i<m_tauCalibFunc.size(); ++i) {
    setOrcaStyleParamsForBin(m_tauCalibFunc.at(i));
  }
}

// The ORCA-style calibration function is a series of inverted quadratic functions 
// (ie x = A + B.y + C.y^2)
//
// This assumes that just one high-energy set of parameters is supplied, together with 
// an energy threshold value (in terms of y, the "true" jet Et). It calculates a set of
// parameters to be applied to lower-Et jets subject to the following constraints:
//  (i) The calibration function is continuous at the high threshold value;
//  (ii) Its slope is also continuous at the high threshold value;
//  (iii) At the low (zero-suppression) threshold, the calibration function returns y=x.
//
void produceTrivialCalibrationLut::setOrcaStyleParamsForBin(std::vector<double>& paramsForBin)
{
  assert (paramsForBin.size() == 4);

  double x2 = paramsForBin.at(0);
  double A  = paramsForBin.at(1);
  double B  = paramsForBin.at(2);
  double C  = paramsForBin.at(3);

  double y2 = 2*(x2-A)/(B + sqrt(B*B - 4*(A-x2)*C));

  double y1 = 5.0; // This was zero suppresion threshold parameter, but such a parameter is no longer defined
  double g = (y1 - (A + y1*(B+y1*C)))/(pow((y2-y1),2));
  A = A + g*y2*y2;
  B = B - 2.0*g*y2;
  C = C + g;

  paramsForBin.push_back(y1);
  paramsForBin.push_back(A);
  paramsForBin.push_back(B);
  paramsForBin.push_back(C);
}
//
//--------------------------------------------------------------------------


