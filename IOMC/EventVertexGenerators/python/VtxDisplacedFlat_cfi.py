import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import FlatVtxDisplacedParameters
VtxSmeared = cms.EDProducer("FlatEvtVtxGenerator",
    FlatVtxDisplacedParameters,
    src = cms.InputTag("generator", "unsmeared"),
)



