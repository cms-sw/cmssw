/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemRPRecHitProducerAlgorithm.h"

//----------------------------------------------------------------------------------------------------

void TotemRPRecHitProducerAlgorithm::buildRecoHits(const edm::DetSet<TotemRPCluster>& input, 
    edm::DetSet<TotemRPRecHit>& output)
{
  for (auto it : input)
  {
    constexpr double nominal_sigma = 0.0191;
    output.push_back(TotemRPRecHit(rp_topology_.GetHitPositionInReadoutDirection(it.getCenterStripPosition()), nominal_sigma));
  }  
}
