/**
 * Author: Seth Cooper, University of Minnesota
 * Created: 21 Mar 2011
 * $Id: EcalTimeOffsetConstant.cc,v 1.1 2011/03/22 16:13:04 argiro Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

EcalTimeOffsetConstant::EcalTimeOffsetConstant() 
{
  EBvalue_=0.;
  EEvalue_=0.;
}

EcalTimeOffsetConstant::EcalTimeOffsetConstant(const float& EBvalue,const float& EEvalue)
{
  EBvalue_ = EBvalue;
  EEvalue_ = EEvalue;
}

EcalTimeOffsetConstant::~EcalTimeOffsetConstant()
{
}
