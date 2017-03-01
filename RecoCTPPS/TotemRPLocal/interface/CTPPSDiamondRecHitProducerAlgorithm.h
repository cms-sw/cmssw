/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_CTPPSDiamondRecHitProducerAlgorithm
#define RecoCTPPS_TotemRPLocal_CTPPSDiamondRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"

class CTPPSDiamondRecHitProducerAlgorithm
{
  public:
    CTPPSDiamondRecHitProducerAlgorithm( const edm::ParameterSet& conf );

    void build( const TotemRPGeometry*, const edm::DetSetVector<CTPPSDiamondDigi>&, edm::DetSetVector<CTPPSDiamondRecHit>& );

  private:
    /// Conversion constant between HPTDC time slice and absolute time (in ns)
    double ts_to_ns_;
    int t_shift_;
};

#endif
