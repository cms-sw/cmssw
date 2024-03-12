import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("HLLHCEvtVtxGenerator",
                            HLLHCCrabKissingVtxSmearingParameters,
                            VtxSmearedCommon
                            )

# foo bar baz
# l6Ha5mNNM3AMa
