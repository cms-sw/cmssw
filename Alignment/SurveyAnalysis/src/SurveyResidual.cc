#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/SurveyAnalysis/interface/SurveyResidual.h"

survey::RotMatrix SurveyResidual::diffMomRot() const
{
  unsigned int nSister = theSisters.size();

  survey::Vectors nominalVs; // nominal points from mother's pos
  survey::Vectors currentVs; // current points from mother's pos

  for (unsigned int i = 0; i < nSister; ++i)
  {
    const Alignable* sis = theSisters[i];

    const survey::Points& nominalPoints = sis->survey()->globalPoints();
    const survey::Points& currentPoints = sis->surface().toGlobal( sis->survey()->localPoints() );

    for (unsigned int j = 0; j < nominalPoints.size(); ++j)
    {
      nominalVs.push_back(nominalPoints[j] - theNominalMomPos);
      currentVs.push_back(currentPoints[j] - theCurrentMomPos);
    }
  }

  return survey::diffRot(currentVs, nominalVs);
}

SurveyResidual::SurveyResidual(const Alignable& ali,
			       AlignableType type,
			       bool bias):
  theSurface( ali.surface() )
{
// Find mother matching given type

  const Alignable* dau = &ali; // start finding from sensor

  while (dau)
  {
    const Alignable* mom = dau->mother();

    if (!mom)
    {
      throw cms::Exception("ConfigError")
	<< "Alignable (id = " << ali.geomDetId().rawId()
	<< ") does not belong to a composite of type " << type
	<< ". Abort!"
	<< std::endl;
    }

    if (mom->alignableObjectId() == type) break; // found

    dau = mom; // move up a level
  }

  findSisters(dau, bias);

  if (theSisters.size() == 0)
  {
    throw cms::Exception("ConfigError")
      << "You are finding an unbiased residual of an alignable "
      << " (id = " << ali.geomDetId().rawId()
      << ") which has no sister. Abort!"
      << std::endl;
  }

  calculate(ali);
}

AlgebraicVector SurveyResidual::sensorResidual() const
{
  LocalVector deltaR = theSurface.toLocal(theCurrentVs[0] - theNominalVs[0]);

// Match the centers of current and nominal surfaces to find the angular
// displacement about the center.

  survey::Vectors nominalVs = theNominalVs;
  survey::Vectors currentVs = theCurrentVs;

  for (unsigned int j = 0; j < nominalVs.size(); ++j)
  {
    nominalVs[j] -= theNominalVs[0]; // move to nominal mother's pos
    currentVs[j] -= theCurrentVs[0]; // move to current mother's pos
  }

  survey::RotMatrix rot = survey::diffRot(currentVs, nominalVs);

  AlgebraicVector deltaW = survey::rotMatrixToAngles( theSurface.toLocal(rot) );

  AlgebraicVector deltaRW(6); // (deltaR, deltaW)

  deltaRW(1) = deltaR.x();
  deltaRW(2) = deltaR.y();
  deltaRW(3) = deltaR.z();
  deltaRW(4) = deltaW(1);
  deltaRW(5) = deltaW(2);
  deltaRW(6) = deltaW(3);

  return deltaRW;
}

std::vector<LocalVector> SurveyResidual::pointsResidual() const
{
  std::vector<LocalVector> residuals;

  unsigned int nPoint = theNominalVs.size();

  residuals.reserve(nPoint);

  for (unsigned int j = 0; j < nPoint; ++j)
  {
    GlobalVector res = theCurrentVs[j] - theNominalVs[j];

    residuals.push_back( theSurface.toLocal(res) );
  }

  return residuals;
}

void SurveyResidual::findSisters(const Alignable* ali,
				 bool bias)
{
  theSisters.clear();

  if (bias) ali->getTerminals(theSisters);

  const Alignable* mother = ali->mother();

  if (mother)
  {
    theSisters.reserve(1000);

    const std::vector<Alignable*>& comp = mother->components();

    unsigned int nComp = comp.size();

    for (unsigned int i = 0; i < nComp; ++i)
    {
      const Alignable* dau = comp[i];

      if (dau != ali) dau->getTerminals(theSisters);
    }
  }
}

void SurveyResidual::calculate(const Alignable& ali)
{
  unsigned int nSister = theSisters.size();

  std::vector<const GlobalPoint*> nominalSisPos; // nominal sisters' pos
  std::vector<const GlobalPoint*> currentSisPos; // current sisters' pos

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
	<< " (id = " << sis->geomDetId().rawId() << "). Abort!"
	<< std::endl;
    }
//     std::cout << "Sister " << i << " of type " << sis->alignableObjectId()
// 	      << " has parameters = " << sis->globalPosition()
// 	      << survey::rotMatrixToAngles(sis->globalRotation()).T()
// 	      << std::endl;
    nominalSisPos.push_back( &survey->position() );
    currentSisPos.push_back( &sis->globalPosition() );
  }

  theNominalMomPos = survey::motherPosition(nominalSisPos);
  theCurrentMomPos = survey::motherPosition(currentSisPos);
//   std::cout << "Nominal mother's position = "   << theNominalMomPos
// 	    << "\nCurrent mother's position = " << theCurrentMomPos
// 	    << std::endl;

  const survey::Points& nominalPoints = ali.survey()->globalPoints();
  const survey::Points& currentPoints = theSurface.toGlobal( ali.survey()->localPoints() );

  survey::RotMatrix toCurrent = diffMomRot();
//   std::cout << "Nominal to current angles = "
// 	    << survey::rotMatrixToAngles(toCurrent).T();

  for (unsigned int j = 0; j < nominalPoints.size(); ++j)
  {
    GlobalVector nv = nominalPoints[j] - theNominalMomPos;

    theNominalVs.push_back( GlobalVector( toCurrent * nv.basicVector() ) );
    theCurrentVs.push_back(currentPoints[j] - theCurrentMomPos);
  }
}
