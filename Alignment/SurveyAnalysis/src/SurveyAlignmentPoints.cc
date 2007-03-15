#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"
#include "Alignment/SurveyAnalysis/interface/SurveyResidual.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"

SurveyAlignmentPoints::SurveyAlignmentPoints(const std::vector<Alignable*>& sensors):
  SurveyAlignment(sensors)
{
}

void SurveyAlignmentPoints::findAlignPars()
{
  unsigned int nSensor = theSensors.size();

  for (unsigned int i = 0; i < nSensor; ++i)
  {
    Alignable* ali = theSensors[i];

    SurveyResidual res1(*ali, AlignableObjectId::AlignablePetal);

    AlgebraicSymMatrix sumJVJT(6, 0); // 6 by 6 symmetric matrix init to 0
    AlgebraicVector    sumJVe(6, 0);  // init to 0

    align::LocalVectors residuals = res1.pointsResidual();

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

    int dummy;
    sumJVJT.invert(dummy);
    sumJVe = -sumJVJT * sumJVe;

    ali->setAlignmentParameters( new SurveyParameters(ali, sumJVe, sumJVJT) );
  }
}
