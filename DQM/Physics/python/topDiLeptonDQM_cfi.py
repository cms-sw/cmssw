import FWCore.ParameterSet.Config as cms

topDiLeptonDQM = cms.EDAnalyzer("TopDiLeptonDQM",
    muonCollection = cms.InputTag('muons'),
    pT_cut  = cms.double( 4.0 ),
    eta_cut = cms.double( 5.0 ),
    moduleName = cms.untracked.string('Top/DiLepton')
)

topDiLeptonAnalyzer = cms.Sequence(topDiLeptonDQM)
