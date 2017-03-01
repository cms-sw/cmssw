/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemRPRecHitProducerAlgorithm
#define RecoCTPPS_TotemRPLocal_TotemRPRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

class TotemRPRecHitProducerAlgorithm
{
  public:
    TotemRPRecHitProducerAlgorithm(const edm::ParameterSet& conf)
    {
    }

    void buildRecoHits(const edm::DetSet<TotemRPCluster>& input, edm::DetSet<TotemRPRecHit>& output);

  private:
    RPTopology rp_topology_;
};

#endif
