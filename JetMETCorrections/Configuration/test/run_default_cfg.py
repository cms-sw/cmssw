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
    fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/START36_V4-v1/0011/FE30B408-D044-DF11-92FC-0026189438C1.root'
    'file:///data/kkousour/FE30B408-D044-DF11-92FC-0026189438C1.root'
    )
)

#!
#! SERVICES
#!
process.TFileService=cms.Service("TFileService",fileName=cms.string('histos.root'))

#!
#! JET CORRECTION
#!
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

process.ak5CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak7CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak7CaloJetsL2L3')
process.kt4CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt4CaloJetsL2L3')
process.kt6CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt6CaloJetsL2L3')
process.ic5CaloL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ic5CaloJetsL2L3')

process.ak5PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5PFJetsL2L3')
process.ak7PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak7PFJetsL2L3')
process.kt4PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt4PFJetsL2L3')
process.kt6PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'kt6PFJetsL2L3')
process.ic5PFL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ic5PFJetsL2L3')

process.ak5JPTL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5JPTJetsL2L3')
process.ak5TrackL2L3Histos = process.ak5CaloL2L3Histos.clone(src = 'ak5TrackJetsL2L3')

#
# RUN!
#
process.run = cms.Path(
#------ create the corrected calojet collection and run the histogram module ------
process.ak5CaloJetsL2L3 * process.ak5CaloL2L3Histos * process.ak7CaloJetsL2L3 * process.ak7CaloL2L3Histos * 
process.kt4CaloJetsL2L3 * process.kt4CaloL2L3Histos * process.kt6CaloJetsL2L3 * process.kt6CaloL2L3Histos * 
process.ic5CaloJetsL2L3 * process.ic5CaloL2L3Histos *
#------ create the corrected pfjet collection and run the histogram module --------
process.ak5PFJetsL2L3 * process.ak5PFL2L3Histos * process.ak7PFJetsL2L3 * process.ak7PFL2L3Histos *
process.kt4PFJetsL2L3 * process.kt4PFL2L3Histos * process.kt6PFJetsL2L3 * process.kt6PFL2L3Histos *
process.ic5PFJetsL2L3 * process.ic5PFL2L3Histos *
#------ create the corrected jptjet collection and run the histogram module -------
process.ak5JPTJetsL2L3 * process.ak5JPTL2L3Histos *
#------ create the corrected trackjet collection and run the histogram module -----
process.ak5TrackJetsL2L3 * process.ak5TrackL2L3Histos
)
