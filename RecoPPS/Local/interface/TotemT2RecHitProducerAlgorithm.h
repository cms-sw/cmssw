/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoPPS_Local_TotemT2RecHitProducerAlgorithm
#define RecoPPS_Local_TotemT2RecHitProducerAlgorithm

#include "RecoPPS/Local/interface/TimingRecHitProducerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"

class TotemT2RecHitProducerAlgorithm : public TimingRecHitProducerAlgorithm<TotemGeometry,
                                                                            edmNew::DetSetVector<TotemT2Digi>,
                                                                            edmNew::DetSetVector<TotemT2RecHit> > {
public:
  using TimingRecHitProducerAlgorithm::TimingRecHitProducerAlgorithm;
  void build(const TotemGeometry&,
             const edmNew::DetSetVector<TotemT2Digi>&,
             edmNew::DetSetVector<TotemT2RecHit>&) override;
};

#endif
