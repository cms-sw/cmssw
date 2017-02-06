import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGSFixLinker_cfi import pfGSFixLinker
from RecoParticleFlow.PFProducer.pfEGFootprintGSFixLinker_cfi import pfEGFootprintGSFixLinker

particleFlowGSFixed = pfGSFixLinker.clone(
    GsfElectrons = 'gedGsfElectronsGSFixed',
    Photons = 'gedPhotonsGSFixed'
)

particleBasedIsolationGSFixed = pfEGFootprintGSFixLinker.clone(
    GsfElectrons = 'gedGsfElectronsGSFixed',
    Photons = 'gedPhotonsGSFixed'
)

from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs
particleFlowTmpPtrs = particleFlowPtrs.clone()
particleFlowPtrs = particleFlowPtrs

particleFlowLinks = cms.Sequence(
    particleFlowGSFixed +
    particleBasedIsolationGSFixed +
    particleFlowTmpPtrs +
    particleFlowPtrs
)
