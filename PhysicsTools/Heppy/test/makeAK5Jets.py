import FWCore.ParameterSet.Config as cms


process = cms.Process("EX")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:../ZLL-8A345C56-6665-E411-9C25-1CC1DE04DF20.root")
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(['drop *'])
)
process.endpath= cms.EndPath(process.OUT)

from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets

# Select candidates that would pass CHS requirements
process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))

#makes chs ak5 jets   (instead of ak4 that are default in miniAOD 70X)
process.ak5PFJetsCHS = ak5PFJets.clone(src = 'chs')
process.OUT.outputCommands.append("keep *_ak5PFJetsCHS_*_EX")

process.options = cms.untracked.PSet( 
        wantSummary = cms.untracked.bool(True), # while the timing of this is not reliable in unscheduled mode, it still helps understanding what was actually run 
        allowUnscheduled = cms.untracked.bool(True)
)
