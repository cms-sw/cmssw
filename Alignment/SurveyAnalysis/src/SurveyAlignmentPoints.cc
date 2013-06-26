#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"

SurveyAlignmentPoints::SurveyAlignmentPoints(const align::Alignables& sensors,
					     const std::vector<align::StructureType>& levels):
  SurveyAlignment(sensors, levels)
{
}

void SurveyAlignmentPoints::findAlignPars(bool bias)
{
  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    AlgebraicSymMatrix sumJVJT(6, 0); // 6 by 6 symmetric matrix init to 0
    AlgebraicVector    sumJVe(6, 0);  // init to 0

    for (unsigned int l = 0; l < theLevels.size(); ++l)
    {
      SurveyResidual res(*ali, theLevels[l], bias);

      if ( !res.valid() ) continue;

      align::LocalVectors residuals = res.pointsResidual();

      unsigned int nPoints = residuals.size();

      for (unsigned int j = 0; j < nPoints; ++j)
      {
	AlgebraicMatrix J = ali->survey()->derivatives(j);
	AlgebraicSymMatrix V(3, 1); // identity for now
	AlgebraicVector e(3); // local residual

	const align::LocalVector& lr = residuals[j];

	e(1) = lr.x(); e(2) = lr.y(); e(3) = lr.z();
	V /= 1e-4 * 1e-4;
	sumJVe  += J * (V * e);
	sumJVJT += V.similarity(J);
      }
    }

    int dummy;
    sumJVJT.invert(dummy); // sumJVJT = sumJVJT^-1
    sumJVe = -sumJVJT * sumJVe;

    ali->setAlignmentParameters( new SurveyParameters(ali, sumJVe, sumJVJT) );
  }
}
