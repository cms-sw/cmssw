/** \file Alignable.cc
 *
 *  $Date: 2011/05/23 20:53:18 $
 *  $Revision: 1.22 $
 *  (last update by $Author: mussgill $)
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
  theAlignmentParameters(0),
  theMother(0),
  theSurvey(0)
{
}

//__________________________________________________________________________________________________
Alignable::Alignable(align::ID id, const RotationType& rot):
  theDetId(), // FIXME: inconsistent with other ctr., cf. above
  theId(id),
  theSurface(PositionType(), rot),
  theCachedSurface(PositionType(), rot),
  theAlignmentParameters(0),
  theMother(0),
  theSurvey(0)
{
}

//__________________________________________________________________________________________________
Alignable::~Alignable()
{
  delete theAlignmentParameters;
  delete theSurvey;
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
  theCachedSurface = theSurface;
  theCachedDisplacement = theDisplacement;
  theCachedRotation = theRotation;
}

void Alignable::restoreCachedTransformation()
{
  // first treat itself
  theSurface = theCachedSurface;
  theDisplacement = theCachedDisplacement;
  theRotation = theCachedRotation;

  // now treat components (a clean design would move that to AlignableComposite...)
  const Alignables comps(this->components());

  for (auto it = comps.begin(); it != comps.end(); ++it) {
    (*it)->restoreCachedTransformation();
  }
 
}

//__________________________________________________________________________________________________
void Alignable::setSurvey( const SurveyDet* survey )
{

  delete theSurvey;
  theSurvey = survey;

}
