/** \file
 *
 *  $Date: 2007/02/03 16:04:09 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/ParametrizedEngine/src/MixedMagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


MixedMagneticField::MixedMagneticField(const MagneticField* param,
				       const MagneticField* full,
				       double scale)
  : theParam(param), theFull(full), theScale(scale) 
{
  
}


GlobalVector MixedMagneticField::inTesla (const GlobalPoint& g) const {
  if (theParam->isDefined(g))  return theParam->inTesla(g);
  return (theFull->inTesla(g))*theScale;
}
