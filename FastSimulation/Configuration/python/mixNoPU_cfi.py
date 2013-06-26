import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.aliases_cfi import simEcalUnsuppressedDigis, simHcalUnsuppressedDigis

from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
simEcalUnsuppressedDigis.hitsProducer = cms.string('famosSimHits')
ecal_digi_parameters.hitsProducer = cms.string('famosSimHits')
ecalDigitizer.hitsProducer = cms.string('famosSimHits')

import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi 
hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
hcalSimBlockFastSim.hitsProducer = cms.string('famosSimHits')
hcalDigitizer = cms.PSet(
    hcalSimBlockFastSim,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

from FastSimulation.Configuration.trackingTruthProducerFastSim_cfi import *
from FastSimulation.Configuration.mixFastSimObjects_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(ecal = cms.PSet(ecalDigitizer),
                          hcal = cms.PSet(hcalDigitizer),
                          mergedtruth = cms.PSet(trackingParticles)),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    checktof = cms.bool(False),                   
    playback = cms.untracked.bool(False),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
    mixObjects = cms.PSet(
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixMuonTracks = cms.PSet(
            mixMuonSimTracks
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)

