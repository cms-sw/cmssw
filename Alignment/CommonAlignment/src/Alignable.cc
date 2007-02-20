/** \file Alignable.cc
 *
 *  $Date: 2007/02/16 16:55:36 $
 *  $Revision: 1.5 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

//__________________________________________________________________________________________________
Alignable::Alignable() : 
  theMisalignmentActive(true), theDetId(0), theAlignmentParameters(0), theMother(0)
{
}


//__________________________________________________________________________________________________
Alignable::~Alignable()
{
  delete theAlignmentParameters;
}


//__________________________________________________________________________________________________
bool Alignable::firstParamComponents(std::vector<Alignable*> &daughts) const
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
      if (!(*iDau)->firstParamComponents(daughts)) {
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
void Alignable::rotateAroundGlobalAxis( const GlobalVector& axis, const float radians )
{

  rotateInGlobalFrame( RotationType(axis.basicVector(),radians) );

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalAxis( const LocalVector& axis, const float radians )
{

  rotateInLocalFrame(RotationType(axis.basicVector(), radians));

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalX( const float radians )
{

  RotationType rot( 1.,  0.,            0.,
		    0.,  cos(radians),  sin(radians),
		    0., -sin(radians),  cos(radians) );

  rotateInGlobalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalX( const float radians )
{
 
  RotationType rot( 1.,  0.,            0.,
		    0.,  cos(radians),  sin(radians),
		    0., -sin(radians),  cos(radians) );

  rotateInLocalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalY( const float radians )
{

  RotationType rot( cos(radians),  0., -sin(radians), 
		    0.,            1.,  0.,
		    sin(radians),  0.,  cos(radians) );

  rotateInGlobalFrame(rot);
  
}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalY( const float radians )
{

  RotationType rot( cos(radians),  0., -sin(radians), 
		    0.,            1.,  0.,
		    sin(radians),  0.,  cos(radians) );
  
  rotateInLocalFrame(rot);

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalZ( const float radians )
{

  RotationType rot(  cos(radians),  sin(radians),  0.,
		    -sin(radians),  cos(radians),  0.,
		     0.,            0.,            1. );

  rotateInGlobalFrame(rot);
  
}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalZ( const float radians)
{

  RotationType rot(  cos(radians),  sin(radians), 0. ,
		    -sin(radians),  cos(radians), 0. ,
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


