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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_8_0_pre7/RelValTTbar/GEN-SIM-RECO/START38_V4-v1/0002/DC19EA07-4286-DF11-BD2B-0030487CD16E.root')
    )

#!
#! SERVICES
#!


#!
#! JET CORRECTION
#!



process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V8::All'
#process.GlobalTag.connect = 'sqlite_file:START38_V6.db'
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionProducersAllAlgos_cff')


process.TFileService=cms.Service("TFileService",fileName=cms.string('histos.root'))

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

process.ak5CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak7CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak7CaloJetsL2L3')
process.kt4CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt4CaloJetsL2L3')
#process.kt6CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt6CaloJetsL2L3')

process.ak5PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5PFJetsL2L3')
process.ak7PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak7PFJetsL2L3')
process.kt4PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt4PFJetsL2L3')
#process.kt6PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt6PFJetsL2L3')

process.ak5JPTL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5JPTJetsL2L3')
process.ak5TrackL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5TrackJetsL2L3')

#
# RUN!
#
process.run = cms.Path(
#------ create the corrected calojet collection and run the histogram module ------
process.ak5CaloJetsL2L3 * process.ak5CaloL2L3Histos * process.ak7CaloJetsL2L3 * process.ak7CaloL2L3Histos * 
process.kt4CaloJetsL2L3 * process.kt4CaloL2L3Histos * 
#process.kt6CaloJetsL2L3 * process.kt6CaloL2L3Histos * 
#------ create the corrected pfjet collection and run the histogram module --------
process.ak5PFJetsL2L3 * process.ak5PFL2L3Histos * process.ak7PFJetsL2L3 * process.ak7PFL2L3Histos *
process.kt4PFJetsL2L3 * process.kt4PFL2L3Histos *
#process.kt6PFJetsL2L3 * process.kt6PFL2L3Histos *
#------ create the corrected jptjet collection and run the histogram module -------
process.ak5JPTJetsL2L3 * process.ak5JPTL2L3Histos *
#------ create the corrected trackjet collection and run the histogram module -----
process.ak5TrackJetsL2L3 * process.ak5TrackL2L3Histos
)








