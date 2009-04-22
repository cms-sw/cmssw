#ifndef PixelCaenAliasBarrel_H
#define PixelCaenAliasBarrel_H

#include "CondFormats/SiPixelObjects/interface/PixelCaenAlias.h"

/** \class PixelCaenAliasBarrel
 *
 *  Helper class to get the PVSS aliases of digital LV, analog LV and biased HV channels
 *  for a given DetId in the Pixel Barrel.
 *
 *  $Date: 2008/11/30 19:41:08 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

class DetId;

class PixelCaenAliasBarrel:
  public PixelCaenAlias
{
  public:

  PixelCaenAliasBarrel(const DetId&);
};

#endif
