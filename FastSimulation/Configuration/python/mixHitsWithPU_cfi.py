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

g4SimHits = cms.EDAlias(
    famosSimHits = cms.VPSet(    cms.PSet(type = cms.string('PCaloHits')),
                                 cms.PSet(type = cms.string('PSimHits')),
                                 cms.PSet(type = cms.string('SimTracks')),
                                 cms.PSet(type = cms.string('SimVertexs'))
                          #mergedtruth = cms.PSet(trackingParticles)),
                          #mergedtruthMuon = cms.PSet(trackingParticlesMuons)), ### comment out for the moment
                                 ),
    MuonSimHits = cms.VPSet(    cms.PSet(type = cms.string('PSimHits'))
                                ),
    g4SimHits = cms.VPSet(    cms.PSet(type = cms.string('PCaloHits'),
                              fromProductInstance = cms.string(''),
                              toProductInstance = cms.string('refined')) )
    )



from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
#simEcalUnsuppressedDigis.hitsProducer = cms.string('g4SimHits')
#ecal_digi_parameters.hitsProducer = cms.string('g4SimHits')
#ecalDigitizer.hitsProducer = cms.string('g4SimHits')

import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi 
hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
#hcalSimBlockFastSim.hitsProducer = cms.string('g4SimHits') 
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
#                                                      fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/giamman/public/MinBias_8TeV_forPileup.root'), # to be substituted with a (future) relval!!!!
                                                      fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/4E330A0D-BA82-E211-9A0A-003048F23D68.root','/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/DEFF70AE-B982-E211-9C0F-003048F1C4B6.root'), # from FullSim
                                                      ),
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
