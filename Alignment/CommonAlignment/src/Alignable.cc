/** \file Alignable.cc
 *
 *  $Date$
 *  $Revision$
 *  (last update by $Author$)
 */

#include "Alignment/CommonAlignment/interface/Alignable.h"

//__________________________________________________________________________________________________
Alignable::Alignable() : 
  theMisalignmentActive(true), theDetId(0), theAlignmentParameters(0), theMother(0)
{
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

  rotateInGlobalFrame( globalRotation().multiplyInverse( rotation*globalRotation() ) );

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundGlobalAxis( const GlobalVector axis, const float radians )
{

  rotateInGlobalFrame( RotationType(axis.basicVector(),radians) );

}


//__________________________________________________________________________________________________
void Alignable::rotateAroundLocalAxis( const LocalVector axis, const float radians )
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


