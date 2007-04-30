/** \file AlignableDetOrUnitPtr
 *
 *  Original author: Gero Flucke, April 2007
 *
 *  $Date: 2007/04/30 11:33:57 $
 *  $Revision: 1.1.2.1 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

// Due to some implications with includes
// (needed for converison from AlignableDet(Unit)* to Alignable*)
// it is currently not possible to inline the following methods in the header...

///////////////////////////////////////////////////////////////////////////////////////////////////
AlignableDetOrUnitPtr::operator Alignable* ()
{ 
  if (theAliDet) return theAliDet;
  else           return theAliDetUnit;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
AlignableDetOrUnitPtr::operator const Alignable* () const 
{
  if (theAliDet) return theAliDet;
  else           return theAliDetUnit;
}

