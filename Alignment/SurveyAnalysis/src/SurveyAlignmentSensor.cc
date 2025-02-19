#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"

SurveyAlignmentSensor::SurveyAlignmentSensor(const align::Alignables& sensors,
					     const std::vector<align::StructureType>& levels):
  SurveyAlignment(sensors, levels)
{
}

void SurveyAlignmentSensor::findAlignPars(bool bias)
{
  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    AlgebraicVector par(6, 0);
    AlgebraicSymMatrix cov(6, 0);

    for (unsigned int l = 0; l < theLevels.size(); ++l)
    {
      SurveyResidual res(*ali, theLevels[l], bias);

      if ( !res.valid() ) continue;

      AlgebraicSymMatrix invCov = res.inverseCovariance();

      par += invCov * res.sensorResidual();
      cov += invCov;
    }

    int dummy;
    cov.invert(dummy); // cov = cov^-1
    par = -cov * par;

    ali->setAlignmentParameters( new SurveyParameters(ali, par, cov) );
  }
}
