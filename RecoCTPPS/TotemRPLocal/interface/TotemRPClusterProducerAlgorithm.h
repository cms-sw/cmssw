/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemRPClusterProducerAlgorithm
#define RecoCTPPS_TotemRPLocal_TotemRPClusterProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TotemDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"

#include <vector>
#include <set>

class TotemRPClusterProducerAlgorithm
{
  public:
    TotemRPClusterProducerAlgorithm(const edm::ParameterSet& param);

    ~TotemRPClusterProducerAlgorithm();
    
    int buildClusters(unsigned int detId, const std::vector<TotemRPDigi> &digi, std::vector<TotemRPCluster> &clusters);
    
  private:
    typedef std::set<TotemRPDigi> TotemRPDigiSet;

    TotemRPDigiSet strip_digi_set_;  ///< input digi set, strip by strip

    const edm::ParameterSet &param_;

    int verbosity_;
};

#endif
