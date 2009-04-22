#ifndef PixelCaenAliasEndcap_H
#define PixelCaenAliasEndcap_H

/** \class PixelCaenAliasEndcap
 *
 *  Helper class to get the PVSS aliases of digital LV, analog LV and biased HV channels
 *  for a given DetId in the Pixel Endcap.
 *
 *  $Date: 2008/11/30 19:41:08 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/SiPixelObjects/interface/PixelCaenAlias.h"

class DetId;

class PixelCaenAliasEndcap:
  public PixelCaenAlias
{
  public:

  PixelCaenAliasEndcap(const DetId&);
};

#endif
