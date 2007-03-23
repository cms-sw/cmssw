#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"

SurveyAlignmentSensor::SurveyAlignmentSensor(const std::vector<Alignable*>& sensors):
  SurveyAlignment(sensors)
{
}

void SurveyAlignmentSensor::findAlignPars()
{
  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    SurveyResidual res1(*ali, AlignableObjectId::AlignablePetal);
    SurveyResidual res2(*ali, AlignableObjectId::AlignableEndcapLayer);
    SurveyResidual res3(*ali, AlignableObjectId::AlignableEndcap);

    AlgebraicSymMatrix invCov1(6, 1); // identity
    AlgebraicSymMatrix invCov2(6, 1); // identity
    AlgebraicSymMatrix invCov3(6, 1); // identity

    invCov1 /= (1e-3 * 1e-3);
    invCov2 /= (1e-3 * 1e-3);
    invCov3 /= (1e-3 * 1e-3);

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
