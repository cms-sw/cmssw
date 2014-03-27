import FWCore.ParameterSet.Config as cms

simEcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
    cms.PSet(type = cms.string('EBDigiCollection')),
    cms.PSet(type = cms.string('EEDigiCollection')),
    cms.PSet(type = cms.string('ESDigiCollection'))
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
    mix = cms.VPSet( cms.PSet(type=cms.string('recoTracks'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('recoTrackExtras'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('TrackingRecHitsOwned'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ) )
    )

from SimGeneral.MixingModule.ecalDigitizer_cfi import *
import SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff
ecal_digi_parameters =SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff.ecal_digi_parameters.clone()
ecal_digi_parameters.hitsProducer = cms.string('famosSimHits')

ecalDigitizer = cms.PSet(ecal_digi_parameters,####FastSim,
                         apd_sim_parameters,
                         ecal_electronics_sim,
                         ecal_cosmics_sim,
                         ecal_sim_parameter_map,
                         ecal_notCont_sim,
                         es_electronics_sim,
                         accumulatorType = cms.string("EcalDigiProducer"),
                         makeDigiSimLinks = cms.untracked.bool(False)
                         )

import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi 
hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
hcalSimBlockFastSim.hitsProducer = cms.string('famosSimHits') 
hcalDigitizer = cms.PSet(
    hcalSimBlockFastSim,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

from FastSimulation.Tracking.recoTrackAccumulator_cfi import *

from FastSimulation.Configuration.mixFastSimObjects_cfi import *

mix = cms.EDProducer("MixingModule",
                     digitizers = cms.PSet(ecal = cms.PSet(ecalDigitizer),
                                           hcal = cms.PSet(hcalDigitizer),
                                           tracker = cms.PSet(trackAccumulator)),
                     LabelPlayback = cms.string(''),
                     maxBunch = cms.int32(0),
                     minBunch = cms.int32(0), ## in terms of 25nsec

                     bunchspace = cms.int32(250), ##ns
                     mixProdStep1 = cms.bool(False),
                     mixProdStep2 = cms.bool(False),

                     #checktof = cms.bool(False),
                     playback = cms.untracked.bool(False),
                     useCurrentProcessOnly = cms.bool(False),
                     
                     input = cms.SecSource("PoolSource",
                         nbPileupEvents = cms.PSet(
                           probFunctionVariable = cms.vint32(0,1), # dummy value, it is replaced by the cfi that imports this file
                           probValue = cms.vdouble(1,0), # dummy value, it is replaced by the cfi that imports this file
                           histoFileName = cms.untracked.string('histProbFunction.root'),
                         ),
                         type = cms.string('probFunction'),
                         sequential = cms.untracked.bool(False),
                         manage_OOT = cms.untracked.bool(False),  ## manage out-of-time pileup
                                           ## setting this to True means that the out-of-time pileup
                                           ## will have a different distribution than in-time, given
                                           ## by what is described on the next line:
##                         OOT_type = cms.untracked.string('Poisson'),  ## generate OOT with a Poisson matching the number chosen for in-time
                                           #OOT_type = cms.untracked.string('fixed'),  ## generate OOT with a fixed distribution
                                           #intFixed_OOT = cms.untracked.int32(2),
                         fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/MinBias_620/MinBias_8TeV_cfi_GEN_SIM_RECO.root'),
                                           #fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/MinBias_620/SingleNuE10_cfi_py_GEN_SIM_RECO.root'),
                                           #fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/user/federica/FastSim/MinBias_620/SingleMuPt10_cfi_py_GEN_SIM_RECO_50evt.root'), 
                                           #fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/relval/CMSSW_7_0_0_pre1/RelValProdMinBias/AODSIM/PRE_ST62_V8-v1/00000/CCA02E69-520F-E311-96CA-003048678BB2.root'), # from FullSim
                                          #### fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/giamman/public/mixing/MinBias_GENSIMRECO.root')
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
