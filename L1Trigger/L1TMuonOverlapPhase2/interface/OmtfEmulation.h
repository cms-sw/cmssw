/*
 * OmtfEmulation.h
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#ifndef L1TMUONOVERLAPPHASE2_OMTFEMULATION_H_
#define L1TMUONOVERLAPPHASE2_OMTFEMULATION_H_

#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFReconstruction.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h"

class OmtfEmulation : public OMTFReconstruction {
public:
  OmtfEmulation(const edm::ParameterSet& edmParameterSet,
                MuStubsInputTokens& muStubsInputTokens,
                edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDTPhPhase2);

  void beginJob();

  ~OmtfEmulation() override;

  void addObservers(const MuonGeometryTokens& muonGeometryTokens,
                    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                    const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) override;

private:
  edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDTPhPhase2;

  unique_ptr<PtAssignmentBase> ptAssignment;
};

#endif /* L1TMUONOVERLAPPHASE2_OMTFEMULATION_H_ */
