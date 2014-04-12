import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.mixFastSimObjects_cfi import *
from FastSimulation.Tracking.recoTrackAccumulator_cfi import *

mixRecoTracks = cms.EDProducer("MixingModule",
                                digitizers = cms.PSet(tracker = cms.PSet(trackAccumulator)),
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
                                                      fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_2_0_pre3-START61_V11/RelValProdMinBias/GEN-SIM-RECO/v1/00000/E86442A7-C182-E211-ABA4-003048F003DC.root'), # relval from FullSim
                                ),
                                mixObjects = cms.PSet(
    mixRealTracks = cms.PSet(
    mixReconstructedTracks
    ),

    )
)
