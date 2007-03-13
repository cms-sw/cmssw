/** \file Alignable.cc
 *
 *  $Date: 2007/03/12 04:04:12 $
 *  $Revision: 1.9 $
 *  (last update by $Author: cklae $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

//__________________________________________________________________________________________________
Alignable::Alignable() : 
  theMisalignmentActive(true), theDetId(0), theAlignmentParameters(0), theMother(0), theSurvey(0)
{
}


//__________________________________________________________________________________________________
Alignable::~Alignable()
{
  delete theAlignmentParameters;
}


//__________________________________________________________________________________________________
void Alignable::deepComponents( std::vector<const Alignable*>& result ) const
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
bool Alignable::firstCompsWithParams(std::vector<Alignable*> &daughts) const
{
  bool isConsistent = true;
  bool hasAliDau = false; // whether there are any (grand-) daughters with parameters
  bool first = true;
  const std::vector<Alignable*> comps(this->components());
  for (std::vector<Alignable*>::const_iterator iDau = comps.begin(), iDauEnd = comps.end();
       iDau != iDauEnd; ++iDau) {
    if ((*iDau)->alignmentParameters()) { // daughter has parameters itself
      daughts.push_back(*iDau);
      if (!first && !hasAliDau) isConsistent = false;
      hasAliDau = true;
    } else {
      const unsigned int nDauBefore = daughts.size();
      if (!(*iDau)->firstCompsWithParams(daughts)) {
        isConsistent = false; // problem down in hierarchy
      }
      if (daughts.size() != nDauBefore) {
        if (!first && !hasAliDau) isConsistent = false;
        hasAliDau = true;
      } else if (hasAliDau) { // no daughters with params, but previous daughters did have daughters
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
AlignmentParameters* Alignable::alignmentParameters() const
{

  return theAlignmentParameters;

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


