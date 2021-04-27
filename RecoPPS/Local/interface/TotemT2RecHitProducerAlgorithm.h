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

#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"

#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

class TotemT2RecHitProducerAlgorithm : public TimingRecHitProducerAlgorithm<TotemGeometry, TotemT2Digi, TotemT2RecHit> {
public:
  using TimingRecHitProducerAlgorithm::TimingRecHitProducerAlgorithm;
  void build(const TotemGeometry&, const edm::DetSetVector<TotemT2Digi>&, edm::DetSetVector<TotemT2RecHit>&) override;
};

#endif
