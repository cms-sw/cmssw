import FWCore.ParameterSet.Config as cms

process = cms.Process("BRs")

from CMGTools.Production.datasetToSource import *
process.source = datasetToSource(
    'gpetrucc',
    '/TTH_Inclusive_M-140_8TeV_pythia6/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM/V5/PAT_CMG_V5_10_0',
   )


## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )

########################################################
#### GEN H->BB VETO
########################################################
process.genHBB = cms.EDFilter("GenParticleSelector",
    src = cms.InputTag("genParticlesPruned"),
    cut = cms.string("pdgId == 25 && (daughterRef(0).pdgId) == 5"),
    filter = cms.bool(True),
)

########################################################
#### GEN-LEVEL SKIMMING
########################################################
process.genHWWZZtt = cms.EDFilter("GenParticleSelector",
    src = cms.InputTag("genParticlesPruned"),
    cut = cms.string("pdgId == 25 && (abs(daughterRef(0).pdgId) == 24 || abs(daughterRef(0).pdgId) == 23 || abs(daughterRef(0).pdgId) == 15)"),
    filter = cms.bool(True),
)
process.genHWW = process.genHWWZZtt.clone(cut = "pdgId == 25 && numberOfDaughters > 0 && (abs(daughterRef(0).pdgId) == 24)")
process.genHZZ = process.genHWWZZtt.clone(cut = "pdgId == 25 && numberOfDaughters > 0 && (abs(daughterRef(0).pdgId) == 23)")
process.genHtt = process.genHWWZZtt.clone(cut = "pdgId == 25 && numberOfDaughters > 0 && (abs(daughterRef(0).pdgId) == 15)")


########################################################
## Below, stuff that you probably don't want to modify
########################################################

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.p_HWW = cms.Path(~process.genHBB + process.genHWW)
process.p_HZZ = cms.Path(~process.genHBB + process.genHZZ)
process.p_Htt = cms.Path(~process.genHBB + process.genHtt)
