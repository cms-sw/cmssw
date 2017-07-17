import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOEcalTestPulsesRawHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    # choose logical OR between Triggerbits
    andOr=True,
    HLTPaths=['HLT_*'],
    # tolerate triggers stated above, but not available
    throw=False
    )

seqALCARECOEcalTestPulsesRaw = cms.Sequence(ALCARECOEcalTestPulsesRawHLT)
