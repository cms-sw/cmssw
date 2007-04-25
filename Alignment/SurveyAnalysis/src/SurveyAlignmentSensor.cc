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
  static const AlignableObjectId::AlignableObjectIdType levels[5] =
    {AlignableObjectId::Panel,
     AlignableObjectId::Blade,
     AlignableObjectId::HalfDisk,
     AlignableObjectId::HalfCylinder,
     AlignableObjectId::PixelEndcap};

  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    AlgebraicVector par(6, 0);
    AlgebraicSymMatrix cov(6, 0);

    for (unsigned int l = 0; l < 5; ++l)
    {
      SurveyResidual res(*ali, levels[l], bias);

      AlgebraicSymMatrix invCov = res.inverseCovariance();

      par += invCov * res.sensorResidual();
      cov += invCov;
    }

    int dummy;
    cov.invert(dummy);
    par = -cov * par;

    ali->setAlignmentParameters( new SurveyParameters(ali, par, cov) );
  }
}
