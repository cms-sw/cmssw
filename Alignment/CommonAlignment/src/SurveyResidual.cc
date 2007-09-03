#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "DataFormats/Math/interface/Matrix.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/SurveyResidual.h"

using namespace align;

SurveyResidual::SurveyResidual(const Alignable& ali,
			       StructureType type,
			       bool bias):
  theSurface( ali.surface() ),
  theMother(0)
{
// Find mother matching given type

  theMother = &ali; // start finding from this alignable

  while (theMother->alignableObjectId() != type)
  {
    theMother = theMother->mother(); // move up a level

    if (!theMother)
    {
      throw cms::Exception("ConfigError")
	<< "Alignable (id = " << ali.geomDetId().rawId()
	<< ") does not belong to a composite of type " << type;
    }
  }

  if ( !theMother->mother() )
  {
    throw cms::Exception("ConfigError")
      << "The type " << type << " does not have a survey residual defined!\n"
      << "You have probably set the highest hierarchy. Choose a lower level.";
  }

  findSisters(theMother, bias);

  if (theSisters.size() == 0)
  {
    throw cms::Exception("ConfigError")
      << "You are finding an unbiased residual of an alignable "
      << " (id = " << ali.geomDetId().rawId()
      << ") which has no sister. Abort!";
  }

  calculate(ali);
}

AlgebraicVector SurveyResidual::sensorResidual() const
{
  align::LocalVector deltaR = theSurface.toLocal(theCurrentVs[0] - theNominalVs[0]);

// Match the centers of current and nominal surfaces to find the angular
// displacement about the center.

  GlobalVectors nominalVs = theNominalVs;
  GlobalVectors currentVs = theCurrentVs;

  for (unsigned int j = 0; j < nominalVs.size(); ++j)
  {
    nominalVs[j] -= theNominalVs[0]; // move to nominal pos
    currentVs[j] -= theCurrentVs[0]; // move to current pos
  }

  RotationType rot = diffRot(nominalVs, currentVs); // frame rotation

  EulerAngles deltaW = toAngles( theSurface.toLocal(rot) );

  AlgebraicVector deltaRW(6); // (deltaR, deltaW)

  deltaRW(1) = deltaR.x();
  deltaRW(2) = deltaR.y();
  deltaRW(3) = deltaR.z();
  deltaRW(4) = deltaW(1);
  deltaRW(5) = deltaW(2);
  deltaRW(6) = deltaW(3);

  return deltaRW;
}

LocalVectors SurveyResidual::pointsResidual() const
{
  LocalVectors residuals;

  unsigned int nPoint = theNominalVs.size();

  residuals.reserve(nPoint);

  for (unsigned int j = 0; j < nPoint; ++j)
  {
    residuals.push_back( theSurface.toLocal(theCurrentVs[j] - theNominalVs[j]) );
  }

  return residuals;
}

AlgebraicSymMatrix SurveyResidual::inverseCovariance() const
{
  AlgebraicSymMatrix copy(ErrorMatrix::kRows);

  for (unsigned int i = 0; i < ErrorMatrix::kRows; ++i)
    for (unsigned int j = 0; j <= i; ++j)
      copy.fast(i + 1, j + 1) = theInverseCovariance(i, j);

  return copy;
}

void SurveyResidual::findSisters(const Alignable* ali,
				 bool bias)
{
  theSisters.clear();
  theSisters.reserve(1000);

  const std::vector<Alignable*>& comp = ali->mother()->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
  {
    const Alignable* dau = comp[i];

    if (dau != ali || bias) dau->deepComponents(theSisters);
//     if (dau != ali || bias) theSisters.push_back(dau);
  }
}

void SurveyResidual::calculate(const Alignable& ali)
{
  unsigned int nSister = theSisters.size();

// First get sisters' positions

  std::vector<const PositionType*> nominalSisPos; // nominal sisters' pos
  std::vector<const PositionType*> currentSisPos; // current sisters' pos

  nominalSisPos.reserve(nSister);
  currentSisPos.reserve(nSister);

  for (unsigned int i = 0; i < nSister; ++i)
  {
    const Alignable* sis    = theSisters[i];
    const SurveyDet* survey = sis->survey();

    if (!survey)
    {
      throw cms::Exception("ConfigError")
	<< "No survey info is found for Alignable "
	<< " (id = " << sis->geomDetId().rawId() << "). Abort!";
    }

    nominalSisPos.push_back( &survey->position() );
    currentSisPos.push_back( &sis->globalPosition() );
  }

// Then find mother's position using sisters' positions

  PositionType nominalMomPos = motherPosition(nominalSisPos);
  PositionType currentMomPos = motherPosition(currentSisPos);

// Now find rotation from nominal mother to current mother

  GlobalVectors nominalSisVs; // nominal sisters' pos from mother's pos
  GlobalVectors currentSisVs; // current sisters' pos from mother's pos

  for (unsigned int i = 0; i < nSister; ++i)
  {
    const Alignable* sis = theSisters[i];

    const GlobalPoints& nominalSisPoints = sis->survey()->globalPoints();
    const GlobalPoints& currentSisPoints = sis->surface().toGlobal( sis->survey()->localPoints() );

    for (unsigned int j = 0; j < nominalSisPoints.size(); ++j)
    {
      nominalSisVs.push_back(nominalSisPoints[j] - *nominalSisPos[i]);
      currentSisVs.push_back(currentSisPoints[j] - *currentSisPos[i]);
//       nominalSisVs.push_back(nominalSisPoints[j] - nominalMomPos);
//       currentSisVs.push_back(currentSisPoints[j] - currentMomPos);
    }
  }

  RotationType toCurrent = diffRot(currentSisVs, nominalSisVs);

// Finally shift and rotate nominal sensor to current sensor

  const SurveyDet* survey = ali.survey();

  if (!survey)
  {
    throw cms::Exception("ConfigError")
      << "No survey info is found for Alignable "
      << " (id = " << ali.geomDetId().rawId() << "). Abort!";
  }

  const GlobalPoints& nominalPoints = survey->globalPoints();
  const GlobalPoints& currentPoints = theSurface.toGlobal( survey->localPoints() );

  for (unsigned int j = 0; j < nominalPoints.size(); ++j)
  {
    align::GlobalVector nv = nominalPoints[j] - nominalMomPos;

    theNominalVs.push_back( align::GlobalVector( toCurrent * nv.basicVector() ) );
    theCurrentVs.push_back(currentPoints[j] - currentMomPos);
  }

// Find the covariance

  const RotationType& currentFrame = ali.globalRotation();

  for ( const Alignable* a = &ali; a != theMother->mother(); a = a->mother() )
  {
    RotationType deltaR = currentFrame * a->survey()->rotation().transposed();

    math::Matrix<6, 6>::type jac; // 6 by 6 Jacobian init to 0

    jac(0, 0) = deltaR.xx(); jac(0, 1) = deltaR.xy(); jac(0, 2) = deltaR.xz();
    jac(1, 0) = deltaR.yx(); jac(1, 1) = deltaR.yy(); jac(1, 2) = deltaR.yz();
    jac(2, 0) = deltaR.zx(); jac(2, 1) = deltaR.zy(); jac(2, 2) = deltaR.zz();
    jac(3, 3) = deltaR.xx(); jac(3, 4) = deltaR.xy(); jac(3, 5) = deltaR.xz();
    jac(4, 3) = deltaR.yx(); jac(4, 4) = deltaR.yy(); jac(4, 5) = deltaR.yz();
    jac(5, 3) = deltaR.zx(); jac(5, 4) = deltaR.zy(); jac(5, 5) = deltaR.zz();

    theInverseCovariance += ROOT::Math::Similarity( jac, a->survey()->errors() );
  }

  if ( !theInverseCovariance.Invert() )
  {
    throw cms::Exception("ConfigError")
      << "Cannot invert survey error of Alignable (id = "  << ali.id()
      << ") of residual type " << theMother->alignableObjectId()
      << theInverseCovariance;
  }
}

// AlgebraicMatrix SurveyResidual::errorTransform(const RotationType& initialFrame,
// 					       const RotationType& currentFrame) const
// {
// //   align::EulerAngles angles = align::toAngles(r);

// //   align::Scalar s1 = std::sin(angles[0]), c1 = std::cos(angles[0]);
// //   align::Scalar s2 = std::sin(angles[1]), c2 = std::cos(angles[1]);
// //   align::Scalar s3 = std::sin(angles[2]), c3 = std::cos(angles[2]);

//   AlgebraicMatrix drdw(9, 3, 0); // 9 by 3 Jacobian init to 0

// //   drdw(1, 1) =  0;
// //   drdw(1, 2) =  -s2 * c3;
// //   drdw(1, 3) =  c2 * -s3;
// //   drdw(2, 1) =  -s1 * s3 + c1 * s2 * c3;
// //   drdw(2, 2) =  s1 * c2 * c3;
// //   drdw(2, 3) =  c1 * c3 - s1 * s2 * s3;
// //   drdw(3, 1) =  c1 * s3 + s1 * s2 * c3;
// //   drdw(3, 2) =  -c1 * c2 * c3;
// //   drdw(3, 3) =  s1 * c3 + c1 * s2 * s3;
// //   drdw(4, 1) =  0;
// //   drdw(4, 2) =  s2 * s3;
// //   drdw(4, 3) =  -c2 * c3;
// //   drdw(5, 1) =  -s1 * c3 - c1 * s2 * s3;
// //   drdw(5, 2) =  -s1 * c2 * s3;
// //   drdw(5, 3) =  c1 * -s3 - s1 * s2 * c3;
// //   drdw(6, 1) =  c1 * c3 - s1 * s2 * s3;
// //   drdw(6, 2) =  c1 * c2 * s3;
// //   drdw(6, 3) =  s1 * -s3 + c1 * s2 * c3;
// //   drdw(7, 1) =  0;
// //   drdw(7, 2) =  c2;
// //   drdw(7, 3) =  0;
// //   drdw(8, 1) =  -c1 * c2;
// //   drdw(8, 2) =  s1 * s2;
// //   drdw(8, 3) =  0;
// //   drdw(9, 1) =  -s1 * c2;
// //   drdw(9, 2) =  c1 * -s2;
// //   drdw(9, 3) =  0;
//   drdw(2, 3) = drdw(6, 1) = drdw(7, 2) =  1;
//   drdw(3, 2) = drdw(4, 3) = drdw(8, 1) = -1;

//   align::RotationType deltaR = initialFrame * currentFrame.transposed();

//   AlgebraicMatrix dRdr(9, 9, 0); // 9 by 9 Jacobian init to 0

//   dRdr(1, 1) = deltaR.xx(); dRdr(1, 2) = deltaR.yx(); dRdr(1, 3) = deltaR.zx();
//   dRdr(2, 1) = deltaR.xy(); dRdr(2, 2) = deltaR.yy(); dRdr(2, 3) = deltaR.zy();
//   dRdr(3, 1) = deltaR.xz(); dRdr(3, 2) = deltaR.yz(); dRdr(3, 3) = deltaR.zz();
//   dRdr(4, 4) = deltaR.xx(); dRdr(4, 5) = deltaR.yx(); dRdr(4, 6) = deltaR.zx();
//   dRdr(5, 4) = deltaR.xy(); dRdr(5, 5) = deltaR.yy(); dRdr(5, 6) = deltaR.zy();
//   dRdr(6, 4) = deltaR.xz(); dRdr(6, 5) = deltaR.yz(); dRdr(6, 6) = deltaR.zz();
//   dRdr(7, 7) = deltaR.xx(); dRdr(7, 8) = deltaR.yx(); dRdr(7, 9) = deltaR.zx();
//   dRdr(8, 7) = deltaR.xy(); dRdr(8, 8) = deltaR.yy(); dRdr(8, 9) = deltaR.zy();
//   dRdr(9, 7) = deltaR.xz(); dRdr(9, 8) = deltaR.yz(); dRdr(9, 9) = deltaR.zz();

// //   align::RotationType R = r * deltaR;

//   AlgebraicMatrix dWdR(3, 9, 0); // 3 by 9 Jacobian init to 0

//   align::Scalar R11 = deltaR.xx(), R21 = deltaR.yx();
//   align::Scalar R31 = deltaR.zx(), R32 = deltaR.zy(), R33 = deltaR.zz();

//   align::Scalar den1 = R32 * R32 + R33 * R33;
//   align::Scalar den3 = R11 * R11 + R21 * R21;

//   dWdR(1, 8) = -R33 / den1; dWdR(1, 9) = R32 / den1;
//   dWdR(2, 7) = 1 / std::sqrt(1 - R31 * R31);
//   dWdR(3, 1) = R21 / den3; dWdR(3, 4) = -R11 / den3;

//   AlgebraicMatrix dPdp(6, 6, 0);

//   dPdp(1, 1) = deltaR.xx(); dPdp(1, 2) = deltaR.xy(); dPdp(1, 3) = deltaR.xz();
//   dPdp(2, 1) = deltaR.yx(); dPdp(2, 2) = deltaR.yy(); dPdp(2, 3) = deltaR.yz();
//   dPdp(3, 1) = deltaR.zx(); dPdp(3, 2) = deltaR.zy(); dPdp(3, 3) = deltaR.zz();

//   dPdp.sub(4, 4, dWdR * dRdr * drdw);

//   return dPdp;
// }
