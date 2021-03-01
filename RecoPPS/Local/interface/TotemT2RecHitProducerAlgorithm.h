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

#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

class TotemT2RecHitProducerAlgorithm : public TimingRecHitProducerAlgorithm<TotemT2Digi, TotemT2RecHit> {
public:
  using TimingRecHitProducerAlgorithm::TimingRecHitProducerAlgorithm;
  void build(const CTPPSGeometry&, const edm::DetSetVector<TotemT2Digi>&, edm::DetSetVector<TotemT2RecHit>&) override;
};

#endif
