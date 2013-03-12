import FWCore.ParameterSet.Config as cms

simEcalUnsuppressedDigis = cms.EDAlias(
    mixSimCaloHits = cms.VPSet(
    cms.PSet(type = cms.string('EBDigiCollection')),
    cms.PSet(type = cms.string('EEDigiCollection')),
    cms.PSet(type = cms.string('ESDigiCollection'))
    )
    )
simHcalUnsuppressedDigis = cms.EDAlias(
    mixSimCaloHits = cms.VPSet(
    cms.PSet(type = cms.string('HBHEDataFramesSorted')),
    cms.PSet(type = cms.string('HFDataFramesSorted')),
    cms.PSet(type = cms.string('HODataFramesSorted')),
    cms.PSet(type = cms.string('ZDCDataFramesSorted'))
    )
    )

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

from FastSimulation.Configuration.mixFastSimObjects_cfi import *

mixSimCaloHits = cms.EDProducer("MixingModule",
                                digitizers = cms.PSet(ecal = cms.PSet(ecalDigitizer),
                                                      hcal = cms.PSet(hcalDigitizer)),
                                LabelPlayback = cms.string(''),
                                maxBunch = cms.int32(0),
                                minBunch = cms.int32(0),
                                bunchspace = cms.int32(25),
                                checktof = cms.bool(False),
                                playback = cms.untracked.bool(False),
                                mixProdStep1 = cms.bool(False),
                                mixProdStep2 = cms.bool(False),
                                input = cms.SecSource("PoolSource",
                                                      type = cms.string('probFunction'),
                                                      nbPileupEvents = cms.PSet(
    probFunctionVariable = cms.vint32(),
    probValue = cms.vdouble(),
    histoFileName = cms.untracked.string('histProbFunction.root'),
    ),
                                                      sequential = cms.untracked.bool(False),
                                                      manage_OOT = cms.untracked.bool(False),  ## manage out-of-time pileup
                                                      ## setting this to True means that the out-of-time pileup
                                                      ## will have a different distribution than in-time, given
                                                      ## by what is described on the next line:
                                                      OOT_type = cms.untracked.string('None'),  ## generate OOT with a Poisson matching the number chosen for in-time
                                                      #OOT_type = cms.untracked.string('fixed'),  ## generate OOT with a fixed distribution
                                                      #intFixed_OOT = cms.untracked.int32(2),
                                                      fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/giamman/public/MinBias_8TeV_forPileup.root'), # to be substituted with a (future) relval!!!!
                                                      ),
                                mixObjects = cms.PSet(
    mixSH = cms.PSet(
    mixSimHits
    ),
    mixCH = cms.PSet(
    mixCaloHits
    ),
    mixMuonTracks = cms.PSet( # remove?
    mixMuonSimTracks
    ),
    mixHepMC = cms.PSet(
    mixHepMCProducts
    )
    )
)
