import FWCore.ParameterSet.Config as cms

process = cms.Process("copy_events")

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring('file:/opt/user/hinzmann/QCDDiJet_Pt50to80_Summer09_RECO_3_1_X.root')
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# Output module configuration
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('QCDDiJet_Pt50to80_Summer09_RECO_3_1_X_10events.root'),
)
process.outpath = cms.EndPath(process.out)

process.module1=cms.EDAnalyzer("Module1",
inputtag1 = cms.InputTag("module3"),
vinputtag1 = cms.untracked.VInputTag(cms.InputTag("module2")),
vinputtag2 = cms.untracked.VInputTag("module4")
)
process.module2=cms.EDAnalyzer("Module2")
process.module3=cms.EDAnalyzer("Module3")
process.module4=cms.EDAnalyzer("Module4")
process.p=cms.Path(process.module1+~process.module2+~process.module3+~process.module4)
