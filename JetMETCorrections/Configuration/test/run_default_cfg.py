import FWCore.ParameterSet.Config as cms
#!
#! PROCESS
#!
process = cms.Process("JEC")

#!
#! INPUT
#!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_5_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V14-v1/0010/7EF83E04-22EE-DE11-8FA1-00261894396A.root')
    )

#!
#! SERVICES
#!
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout         = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
    )
process.TFileService=cms.Service("TFileService",fileName=cms.string('histos.root'))

#!
#! JET CORRECTION
#!
from JetMETCorrections.Configuration.JetCorrectionEra_cff import *
JetCorrectionEra.era = 'Summer09_7TeV_ReReco332'
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

#!
#! MAKE SOME HISTOGRAMS
#!
jetPtHistogram = cms.PSet(min          = cms.untracked.double(     50),
                          max          = cms.untracked.double(    500),
                          nbins        = cms.untracked.int32 (     50),
                          name         = cms.untracked.string('JetPt'),
                          description  = cms.untracked.string(     ''),
                          plotquantity = cms.untracked.string(   'pt')
                          )

process.ak5CaloHistos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5CaloJets'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('L2L3CorJetAK5Calo'),
    histograms = cms.VPSet(jetPtHistogram)
    )

#
# RUN!
#
process.run = cms.Path(process.L2L3CorJetAK5Calo * process.ak5CaloHistos * process.ak5CaloL2L3Histos)


