/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "DataFormats/TotemRPReco/interface/TotemRPRecHit.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"
#include "Alignment/RPTrackBased/interface/HitCollection.h"

#include <cmath>

Hit::Hit(const TotemRPRecHit &rh) : 
  id(TotemRPDetId::rawToDecId(rh.DetId())),
  position(rh.Position()),
  sigma(rh.Sigma())
{
  // prevents null uncertainty, this would make certain algorithms crash
  if (sigma < 1E-6)
    sigma = 66E-3/sqrt(12.);
}

