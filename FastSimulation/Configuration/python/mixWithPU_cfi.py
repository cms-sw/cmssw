import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.mixFastSimObjects_cfi import *
#from FastSimulation.Configuration.mixOnlyGenParticles_cfi import *

mixGenPU = cms.EDProducer("MixingModule",
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
                          fileNames = cms.untracked.vstring('/store/relval/CMSSW_5_2_1/RelValProdMinBias/GEN-SIM-RAW/START52_V4-v1/0003/C44AE937-6B72-E111-814F-003048F11112.root',
                                                            '/store/relval/CMSSW_5_2_1/RelValProdMinBias/GEN-SIM-RAW/START52_V4-v1/0003/4A699C7B-9872-E111-B6F5-002481E0D7D8.root',
                                                            '/store/relval/CMSSW_5_2_1/RelValProdMinBias/GEN-SIM-RAW/START52_V4-v1/0003/988B34AB-6A72-E111-92CB-002481E0CC00.root'), # these files are stored at CERN; if you are running elsewhere, or these files are not available anymore, you can generate your own minimum bias events to be fed as input to the mixing module, with the following command: cmsDriver.py MinBias_8TeV_cfi  --conditions auto:startup -s GEN --datatier GEN -n 10000  --eventcontent RAWSIM 
                          ),
    mixObjects = cms.PSet(
#        mixSH = cms.PSet(
#            mixSimHits
#        ),
#        mixVertices = cms.PSet(
#            mixSimVertices
#        ),
#        mixCH = cms.PSet(
#            mixCaloHits
#        ),
#        mixMuonTracks = cms.PSet(
#            mixMuonSimTracks
#        ),
#        mixTracks = cms.PSet(
#            mixSimTracks
#        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)


