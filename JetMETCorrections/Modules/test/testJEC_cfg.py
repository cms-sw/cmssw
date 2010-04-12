import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.load('RecoJets.Configuration.RecoJPTJets_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100)
        )

#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///data/kkousour/F07F5DC7-4E45-DF11-8541-003048679188.root')
)

process.TFileService=cms.Service("TFileService",fileName=cms.string('histos.root'))
jetPtHistogram = cms.PSet(min          = cms.untracked.double(     10),
                          max          = cms.untracked.double(    200),
                          nbins        = cms.untracked.int32 (     50),
                          name         = cms.untracked.string('JetPt'),
                          description  = cms.untracked.string(     ''),
                          plotquantity = cms.untracked.string(   'pt')
                          )
process.ak5CaloJetsL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5PFJetsL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5PFJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5TrackJetsL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5TrackJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5JPTJetsL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5JPTJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5JPTJets = process.JetPlusTrackZSPCorJetAntiKt5.clone()
#
# RUN!
#
process.run = cms.Path(
# create the corrected calojet collection and run the histogram module
process.ak5CaloJetsL2L3 * process.ak5CaloJetsL2L3Histos *
# create the corrected pfjet collection and run the histogram module
process.ak5PFJetsL2L3 * process.ak5PFJetsL2L3Histos *
# create the corrected trackjet collection and run the histogram module
process.ak5TrackJetsL2L3 * process.ak5TrackJetsL2L3Histos *
# create the corrected jptjet collection and run the histogram module
process.recoJPTJets * process.ak5JPTJets * process.ak5JPTJetsL2L3 * process.ak5JPTJetsL2L3Histos
)










