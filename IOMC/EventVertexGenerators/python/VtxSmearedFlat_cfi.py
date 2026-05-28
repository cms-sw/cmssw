import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import FlatVtxSmearingParameters
VtxSmeared = cms.EDProducer("FlatEvtVtxGenerator",
    FlatVtxSmearingParameters,
    src = cms.InputTag("generator", "unsmeared"),
)



