import FWCore.ParameterSet.Config as cms

simEcalUnsuppressedDigis = cms.EDAlias( #remove?
    mixSimCaloHits = cms.VPSet(
    cms.PSet(type = cms.string('EBDigiCollection')),
    cms.PSet(type = cms.string('EEDigiCollection')),
    cms.PSet(type = cms.string('ESDigiCollection'))
    )
    )

simEcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
    cms.PSet(type = cms.string('EBDigiCollection')),
    cms.PSet(type = cms.string('EEDigiCollection')),
    cms.PSet(type = cms.string('ESDigiCollection'))
    )
    )

simHcalUnsuppressedDigis = cms.EDAlias(#remove?
    mixSimCaloHits = cms.VPSet(
    cms.PSet(type = cms.string('HBHEDataFramesSorted')),
    cms.PSet(type = cms.string('HFDataFramesSorted')),
    cms.PSet(type = cms.string('HODataFramesSorted')),
    cms.PSet(type = cms.string('HcalUpgradeDataFramesSorted')),
    cms.PSet(type = cms.string('ZDCDataFramesSorted'))
    )
    )

simHcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
    cms.PSet(type = cms.string('HBHEDataFramesSorted')),
    cms.PSet(type = cms.string('HFDataFramesSorted')),
    cms.PSet(type = cms.string('HODataFramesSorted')),
    cms.PSet(type = cms.string('HcalUpgradeDataFramesSorted')),
    cms.PSet(type = cms.string('ZDCDataFramesSorted'))
    )
    )

generalTracks = cms.EDAlias(
    mix = cms.VPSet( cms.PSet(type=cms.string('recoTracks') ) )
    )

from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *

import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi 
hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
hcalSimBlockFastSim.hitsProducer = cms.string('famosSimHits') 
hcalDigitizer = cms.PSet(
    hcalSimBlockFastSim,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

from FastSimulation.Tracking.recoTrackAccumulator_cfi import *

from FastSimulation.Configuration.mixFastSimObjects_cfi import *

mixSimCaloHits = cms.EDProducer("MixingModule", #remove?
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
                                                      fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/MinBias_620/MinBias_8TeV_cfi_GEN_SIM_RECO.root'),
                                                      
                                                      #fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/4E330A0D-BA82-E211-9A0A-003048F23D68.root',
                                                      #  '/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/DEFF70AE-B982-E211-9C0F-003048F1C4B6.root'), # from FullSim
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


mix = cms.EDProducer("MixingModule",
                     digitizers = cms.PSet(ecal = cms.PSet(ecalDigitizer),
                                           hcal = cms.PSet(hcalDigitizer),
                                           tracker = cms.PSet(trackAccumulator)),
                     LabelPlayback = cms.string(''),
                     maxBunch = cms.int32(10),
                     minBunch = cms.int32(0),
                     bunchspace = cms.int32(50),
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
                                           fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/MinBias_620/MinBias_8TeV_cfi_GEN_SIM_RECO.root'),
                                          # fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/ForTuning_SinglePiPt100_newPFpatch/SinglePiPt100_cfi_GEN_FASTSIM_HLT_01.root'),
                                                    
                                           #fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/4E330A0D-BA82-E211-9A0A-003048F23D68.root',
                                           #                                 '/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RAW/v1/00000/DEFF70AE-B982-E211-9C0F-003048F1C4B6.root'), # from FullSim
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
    ),
    mixRecoTracks = cms.PSet(
    mixReconstructedTracks
    )
    )
)
