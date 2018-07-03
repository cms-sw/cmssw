import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
HLLHCCrabKissingVtxSmearingParameters14TeV=HLLHCCrabKissingVtxSmearingParameters.clone(EprotonInGeV = cms.double(7000.0))
VtxSmeared = cms.EDProducer("HLLHCEvtVtxGenerator",
                            HLLHCCrabKissingVtxSmearingParameters14TeV,
                            VtxSmearedCommon
                            )

