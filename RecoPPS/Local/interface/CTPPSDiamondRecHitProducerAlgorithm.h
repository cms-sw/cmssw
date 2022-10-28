/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoPPS_Local_CTPPSDiamondRecHitProducerAlgorithm
#define RecoPPS_Local_CTPPSDiamondRecHitProducerAlgorithm

#include "RecoPPS/Local/interface/TimingRecHitProducerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

class CTPPSDiamondRecHitProducerAlgorithm
    : public TimingRecHitProducerAlgorithm<CTPPSGeometry,
                                           edm::DetSetVector<CTPPSDiamondDigi>,
                                           edm::DetSetVector<CTPPSDiamondRecHit> > {
public:
  using TimingRecHitProducerAlgorithm::TimingRecHitProducerAlgorithm;
  void build(const CTPPSGeometry&,
             const edm::DetSetVector<CTPPSDiamondDigi>&,
             edm::DetSetVector<CTPPSDiamondRecHit>&) override;

private:
  static constexpr unsigned short MAX_CHANNEL = 20;
};

#endif
