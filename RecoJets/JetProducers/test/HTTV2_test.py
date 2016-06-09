########################################
# Imports/Setup
########################################

import sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(['root://xrootd.ba.infn.it//store/mc/RunIISpring16MiniAODv1/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/0899BDA9-AE01-E611-A239-008CFA05EA2C.root']))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.OUT = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
                               outputCommands = cms.untracked.vstring(['drop *']))
process.endpath= cms.EndPath(process.OUT)

# Let CMSSW take care of scheduling 
process.options = cms.untracked.PSet(     
    wantSummary = cms.untracked.bool(True),
    allowUnscheduled = cms.untracked.bool(True)
)


########################################
# HEPTopTagger V2
########################################

# CHS
process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))  

# Jet Clustering Defaults
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.PFJetParameters_cfi import *

# Schedule HEPTopTagger
process.looseOptRHTT = cms.EDProducer(
    "HTTTopJetProducer",
    PFJetParameters.clone(
        src               = cms.InputTag("chs"),
        doAreaFastjet     = cms.bool(True),
        doRhoFastjet      = cms.bool(False),
        jetPtMin          = cms.double(200.0)
    ),
    AnomalousCellParameters,
    useExplicitGhosts = cms.bool(True),
    algorithm           = cms.int32(1),
    jetAlgorithm        = cms.string("CambridgeAachen"),
    rParam              = cms.double(1.5),
    optimalR            = cms.bool(True),
    qJets               = cms.bool(False),
    minFatjetPt         = cms.double(200.),
    minSubjetPt         = cms.double(0.),
    minCandPt           = cms.double(0.),
    maxFatjetAbsEta     = cms.double(99.),
    subjetMass          = cms.double(30.),
    muCut               = cms.double(0.8),
    filtR               = cms.double(0.3),
    filtN               = cms.int32(5),
    mode                = cms.int32(4),
    minCandMass         = cms.double(0.),
    maxCandMass         = cms.double(999999.),
    massRatioWidth      = cms.double(999999.),
    minM23Cut           = cms.double(0.),
    minM13Cut           = cms.double(0.),
    maxM13Cut           = cms.double(999999.),
    writeCompound       = cms.bool(True),
    jetCollInstanceName = cms.string("SubJets")
)


process.OUT.outputCommands.append("keep *_looseOptRHTT_*_Demo")
        



