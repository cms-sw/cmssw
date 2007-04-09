#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"

SurveyAlignmentSensor::SurveyAlignmentSensor(const std::vector<Alignable*>& sensors):
  SurveyAlignment(sensors)
{
}

void SurveyAlignmentSensor::findAlignPars(bool bias)
{
  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    SurveyResidual res1(*ali, AlignableObjectId::AlignablePetal, bias);
    SurveyResidual res2(*ali, AlignableObjectId::AlignableEndcapLayer, bias);
    SurveyResidual res3(*ali, AlignableObjectId::AlignableEndcap, bias);

    AlgebraicSymMatrix invCov1 = res1.inverseCovariance();
    AlgebraicSymMatrix invCov2 = res2.inverseCovariance();
    AlgebraicSymMatrix invCov3 = res3.inverseCovariance();

    AlgebraicVector pars = invCov1 * res1.sensorResidual();
    AlgebraicSymMatrix cov = invCov1;

    pars += invCov2 * res2.sensorResidual();
    cov += invCov2;
    pars += invCov3 * res3.sensorResidual();
    cov += invCov3;

    int dummy;
    cov.invert(dummy);
    pars = -cov * pars;

    ali->setAlignmentParameters( new SurveyParameters(ali, pars, cov) );
  }
}
