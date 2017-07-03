/** \file Alignable.cc
 *
 *  $Date: 2012/12/11 19:11:02 $
 *  $Revision: 1.23 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

//__________________________________________________________________________________________________
Alignable::Alignable(align::ID id, const AlignableSurface& surf):
  theDetId(id), // FIXME: inconsistent with other ctr., but needed for AlignableNavigator 
  theId(id),    // (finally get rid of one of the IDs!)
  theSurface(surf),
  theCachedSurface(surf),
  theAlignmentParameters(nullptr),
  theMother(nullptr),
  theSurvey(nullptr)
{
}

//__________________________________________________________________________________________________
Alignable::Alignable(align::ID id, const RotationType& rot):
  theDetId(), // FIXME: inconsistent with other ctr., cf. above
  theId(id),
  theSurface(PositionType(), rot),
  theCachedSurface(PositionType(), rot),
  theAlignmentParameters(nullptr),
  theMother(nullptr),
  theSurvey(nullptr)
{
}

//__________________________________________________________________________________________________
Alignable::~Alignable()
{
  delete theAlignmentParameters;
  delete theSurvey;
}

//__________________________________________________________________________________________________
void Alignable::update(align::ID id, const AlignableSurface& surf)
{
  if (theId != id) {
    throw cms::Exception("Alignment")
      << "@SUB=Alignable::update\n"
      << "Current alignable ID does not match ID of the update.";
  }
  const auto shift = surf.position() - theSurface.position();
  theSurface = surf;

  // reset displacement and rotations after update
  theDisplacement = GlobalVector();
  theRotation = RotationType();

  // recalculate containing composite's position
  updateMother(shift);
}

//__________________________________________________________________________________________________
bool Alignable::firstCompsWithParams(Alignables &paramComps) const
{
  bool isConsistent = true;
  bool hasAliComp = false; // whether there are any (grand-) daughters with parameters
  bool first = true;
  const Alignables comps(this->components());
  for (Alignables::const_iterator iComp = comps.begin(), iCompEnd = comps.end();
       iComp != iCompEnd; ++iComp) {
    if ((*iComp)->alignmentParameters()) { // component has parameters itself
      paramComps.push_back(*iComp);
      if (!first && !hasAliComp) isConsistent = false;
      hasAliComp = true;
    } else {
      const unsigned int nCompBefore = paramComps.size();
      if (!(*iComp)->firstCompsWithParams(paramComps)) {
        isConsistent = false; // problem down in hierarchy
      }
      if (paramComps.size() != nCompBefore) {
        if (!first && !hasAliComp) isConsistent = false;
        hasAliComp = true;
      } else if (hasAliComp) { // no components with params, but previous component did have comps.
        isConsistent = false;
      }
    }
    first = false;
  }

  return isConsistent;
}

//__________________________________________________________________________________________________
bool Alignable::lastCompsWithParams(Alignables& paramComps) const
{
  bool isConsistent = true;
  bool hasAliComp = false;
  bool first = true;
  const Alignables comps(this->components());
  for (const auto& iComp: comps) {
    const auto nCompsBefore = paramComps.size();
    isConsistent = iComp->lastCompsWithParams(paramComps);
    if (paramComps.size() == nCompsBefore) {
      if (iComp->alignmentParameters()) {
	paramComps.push_back(iComp);
	if (!first && !hasAliComp) isConsistent = false;
	hasAliComp = true;
      }
    } else {
      if (hasAliComp) {
	isConsistent = false;
      }
      if (!first && !hasAliComp) isConsistent = false;
      hasAliComp = true;
    }
    first = false;
  }

  return isConsistent;
}

//__________________________________________________________________________________________________
void Alignable::setAlignmentParameters( AlignmentParameters* dap )
{

  delete theAlignmentParameters;
  theAlignmentParameters = dap;

}


//__________________________________________________________________________________________________
void Alignable::rotateInLocalFrame( const RotationType& rotation)
{

  // This is done by simply transforming the rotation from
  // the local system O to the global one  O^-1 * Rot * O
  // and then applying the global rotation  O * Rot

  rotateInGlobalFrame( surface().toGlobal(rotation) );

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalAxis( const GlobalVector& axis, Scalar radians )
{

  rotateInGlobalFrame( RotationType(axis.basicVector(),radians) );

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalAxis( const LocalVector& axis, Scalar radians )
{

  rotateInLocalFrame(RotationType(axis.basicVector(), radians));

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalX( Scalar radians )
{

  RotationType rot( 1.,  0.,            0.,
		    0.,  std::cos(radians),  std::sin(radians),
		    0., -std::sin(radians),  std::cos(radians) );

  rotateInGlobalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalX( Scalar radians )
{
 
  RotationType rot( 1.,  0.,            0.,
		    0.,  std::cos(radians),  std::sin(radians),
		    0., -std::sin(radians),  std::cos(radians) );

  rotateInLocalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalY( Scalar radians )
{

  RotationType rot( std::cos(radians),  0., -std::sin(radians), 
		    0.,            1.,  0.,
		    std::sin(radians),  0.,  std::cos(radians) );

  rotateInGlobalFrame(rot);
  
}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalY( Scalar radians )
{

  RotationType rot( std::cos(radians),  0., -std::sin(radians), 
		    0.,            1.,  0.,
		    std::sin(radians),  0.,  std::cos(radians) );
  
  rotateInLocalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalZ( Scalar radians )
{

  RotationType rot(  std::cos(radians),  std::sin(radians),  0.,
		    -std::sin(radians),  std::cos(radians),  0.,
		     0.,            0.,            1. );

  rotateInGlobalFrame(rot);
  
}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalZ( Scalar radians)
{

  RotationType rot(  std::cos(radians),  std::sin(radians), 0. ,
		    -std::sin(radians),  std::cos(radians), 0. ,
		     0.,            0.,           1. );
  
  rotateInLocalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::addDisplacement( const GlobalVector& displacement )
{

  theDisplacement += displacement;

}

//__________________________________________________________________________________________________
void Alignable::addRotation( const RotationType& rotation ) 
{

  theRotation *= rotation;

}

//__________________________________________________________________________________________________
AlignmentSurfaceDeformations* Alignable::surfaceDeformations( void ) const
{

  typedef std::pair<int,SurfaceDeformation*> IdSurfaceDeformationPtrPair;

  std::vector<IdSurfaceDeformationPtrPair> result;
  surfaceDeformationIdPairs(result);
  std::sort( result.begin(), 
 	     result.end(), 
	     lessIdAlignmentPair<IdSurfaceDeformationPtrPair>() );
  
  AlignmentSurfaceDeformations* allSurfaceDeformations = new AlignmentSurfaceDeformations();
  
  for ( std::vector<IdSurfaceDeformationPtrPair>::const_iterator iPair = result.begin();
	iPair != result.end();
	++iPair) {
    
    // should we check for 'empty' parameters here (all zeros) and skip ?
    // may be add 'empty' method to SurfaceDeformation
    allSurfaceDeformations->add((*iPair).first,
				(*iPair).second->type(),
				(*iPair).second->parameters());
  }
  
  return allSurfaceDeformations;

}

void Alignable::cacheTransformation()
{
  // first treat itself
  theCachedSurface = theSurface;
  theCachedDisplacement = theDisplacement;
  theCachedRotation = theRotation;

  // now treat components (a clean design would move that to AlignableComposite...)
  const Alignables comps(this->components());

  for (auto it = comps.begin(); it != comps.end(); ++it) {
    (*it)->cacheTransformation();
  }

}

void Alignable::cacheTransformation(const align::RunNumber& run)
{
  // first treat itself
  surfacesCache_[run] = theSurface;
  displacementsCache_[run] = theDisplacement;
  rotationsCache_[run] = theRotation;

  // now treat components (a clean design would move that to AlignableComposite...)
  const Alignables comps(this->components());
  for (auto& it: comps) it->cacheTransformation(run);
}

void Alignable::restoreCachedTransformation()
{
  // first treat itself
  theSurface = theCachedSurface;
  theDisplacement = theCachedDisplacement;
  theRotation = theCachedRotation;

  // now treat components (a clean design would move that to AlignableComposite...)
  const auto comps = this->components();

  for (auto it = comps.begin(); it != comps.end(); ++it) {
    (*it)->restoreCachedTransformation();
  }
 
}

void Alignable::restoreCachedTransformation(const align::RunNumber& run)
{
  if (surfacesCache_.find(run) == surfacesCache_.end()) {
    throw cms::Exception("Alignment")
      << "@SUB=Alignable::restoreCachedTransformation\n"
      << "Trying to restore cached transformation for a run (" << run
      << ") that has not been cached.";
  } else {
    // first treat itself
    theSurface = surfacesCache_[run];
    theDisplacement = displacementsCache_[run];
    theRotation = rotationsCache_[run];

    // now treat components (a clean design would move that to AlignableComposite...)
    const auto comps = this->components();
    for (auto it: comps) it->restoreCachedTransformation();
  }
}

//__________________________________________________________________________________________________
void Alignable::setSurvey( const SurveyDet* survey )
{

  delete theSurvey;
  theSurvey = survey;

}

//______________________________________________________________________________
void Alignable::updateMother(const GlobalVector& shift) {

  if (!theMother) return;

  const auto thisComps = this->deepComponents().size();
  const auto motherComps = theMother->deepComponents().size();
  const auto motherShift = shift * static_cast<Scalar>(thisComps) / motherComps;

  switch(theMother->compConstraintType()) {
  case CompConstraintType::NONE:
    break;
  case CompConstraintType::POSITION_Z:
    theMother->theSurface.move(GlobalVector(0,0, motherShift.z()));
    theMother->updateMother(GlobalVector(0,0, motherShift.z()));
    break;
  case CompConstraintType::POSITION:
    theMother->theSurface.move(motherShift);
    theMother->updateMother(motherShift);
    break;
  }
}
