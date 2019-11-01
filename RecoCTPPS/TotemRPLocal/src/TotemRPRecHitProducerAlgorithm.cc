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
                                                   edm::DetSet<TotemRPRecHit>& output) {
  for (edm::DetSet<TotemRPCluster>::const_iterator it = input.begin(); it != input.end(); ++it) {
    constexpr double nominal_sigma = 0.0191;
    output.push_back(
        TotemRPRecHit(rp_topology_.GetHitPositionInReadoutDirection(it->centerStripPosition()), nominal_sigma));
  }
}
