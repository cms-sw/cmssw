import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOPromptCalibProdSiStripHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'PromptCalibProdSiStrip',
    throw = False # tolerate triggers stated above, but not available
    )
                     
seqALCARECOPromptCalibProdSiStrip = cms.Sequence(ALCARECOPromptCalibProdSiStripHLT)


