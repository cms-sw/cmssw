## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:patTuple_isoval.root'
    )
)

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzerIsolation.root')
)

process.CITKIso = cms.EDAnalyzer("IsolationAnalysis",
    # input collection
    muonLabel = cms.InputTag("muons"),
    IsoValMuonsCITK = cms.VInputTag(
                                    cms.InputTag( "muonPFNoPileUpIsolation:h+-DR030-ThresholdVeto000-ConeVeto000"),
                                    cms.InputTag( "muonPFNoPileUpIsolation:h0-DR030-ThresholdVeto050-ConeVeto000"),
                                    cms.InputTag( "muonPFNoPileUpIsolation:gamma-DR030-ThresholdVeto050-ConeVeto000"),
                                    cms.InputTag( "muonPFPileUpIsolation:h+-DR030-ThresholdVeto050-ConeVeto000")
                                   ),
    IsoValMuonsIsoDeposit = cms.VInputTag(
                                    cms.InputTag( "muPFIsoValueCharged03"),
                                    cms.InputTag( "muPFIsoValueNeutral03"),
                                    cms.InputTag( "muPFIsoValueGamma03"),
                                    cms.InputTag( "muPFIsoValuePU03")
                                   )
)

process.p = cms.Path(
            process.CITKIso
)



