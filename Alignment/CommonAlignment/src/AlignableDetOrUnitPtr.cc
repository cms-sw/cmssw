/** \file AlignableDetOrUnitPtr
 *
 *  Original author: Gero Flucke, April 2007
 *
 *  $Date: 2009/02/28 21:05:00 $
 *  $Revision: 1.3 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

// Due to some implications with includes
// (needed for converison from AlignableDet(Unit)* to Alignable*)
// it is currently not possible to inline the following methods in the header...

///////////////////////////////////////////////////////////////////////////////////////////////////
AlignableDetOrUnitPtr::operator Alignable* ()
{ 
  if (theAliBeamSpot) return theAliBeamSpot;
  else if (theAliDet) return theAliDet;
  else                return theAliDetUnit;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
AlignableDetOrUnitPtr::operator const Alignable* () const 
{
  if (theAliBeamSpot) return theAliBeamSpot;
  else if (theAliDet) return theAliDet;
  else                return theAliDetUnit;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
const AlignmentPositionError* AlignableDetOrUnitPtr::alignmentPositionError() const
{
  if (theAliBeamSpot)     return theAliBeamSpot->alignmentPositionError();
  else if (theAliDet)     return theAliDet->alignmentPositionError();
  else if (theAliDetUnit) return theAliDetUnit->alignmentPositionError();
  else                    return nullptr;
}

