/** \file Alignable.cc
 *
 *  $Date: 2007/06/21 16:18:28 $
 *  $Revision: 1.14 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

//__________________________________________________________________________________________________
Alignable::Alignable(const GeomDet* det):
  theDetId( det->geographicalId() ),
  theId( theDetId.rawId() ),
  theSurface( det->surface() ),
  theAlignmentParameters(0),
  theMother(0),
  theSurvey(0)
{
}


Alignable::Alignable(uint32_t id, const RotationType& rot):
  theId(id),
  theSurface(PositionType(), rot),
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
void Alignable::deepComponents( std::vector<const Alignable*>& result ) const
{
  const Alignables& comp = components();

  unsigned int nComp = comp.size();

  if (nComp > 0)
    for (unsigned int i = 0; i < nComp; ++i)
    {
      comp[i]->deepComponents(result);
    }
  else
    result.push_back(this);
}

//__________________________________________________________________________________________________
void Alignable::deepComponents( std::vector<Alignable*>& result )
{
  const std::vector<Alignable*>& comp = components();

  unsigned int nComp = comp.size();

  if (nComp > 0)
    for (unsigned int i = 0; i < nComp; ++i)
    {
      comp[i]->deepComponents(result);
    }
  else
    result.push_back(this);
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
void Alignable::setSurvey( const SurveyDet* survey )
{

  delete theSurvey;
  theSurvey = survey;

}
