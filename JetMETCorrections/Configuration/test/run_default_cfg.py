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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_2_0_pre7/RelValProdTTbar/GEN-SIM-RECO/PRE_STA72_V4-v1/00000/B223AEC2-B94B-E411-884B-00261894395F.root')
    )

#!
#! SERVICES
#!


#!
#! JET CORRECTION
#!



process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup')
process.load('JetMETCorrections.Configuration.CorrectedJetProducersDefault_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducersAllAlgos_cff')


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

process.ak4CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak4CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak7CaloL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak7CaloJetsL2L3')
process.kt4CaloL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'kt4CaloJetsL2L3')
#process.kt6CaloL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'kt6CaloJetsL2L3')

process.ak4PFL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak4PFJetsL2L3')
process.ak4PFCHSL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak4PFCHSJetsL2L3')
process.ak8PFL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak8PFJetsL2L3')
process.kt4PFL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'kt4PFJetsL2L3')
#process.kt6PFL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'kt6PFJetsL2L3')

process.ak4JPTL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak4JPTJetsL2L3')
process.ak4TrackL2L3Histos = process.ak4CaloL2L3Histos.clone(src = 'ak4TrackJetsL2L3')

#
# RUN!
#
process.run = cms.Path(
#------ create the corrected calojet collection and run the histogram module ------
#process.ak4CaloL2L3CorrectorChain * process.ak4CaloJetsL2L3 * process.ak4CaloL2L3Histos *
#process.ak7CaloL2L3CorrectorChain * process.ak7CaloJetsL2L3 * process.ak7CaloL2L3Histos *
#process.kt4CaloL2L3CorrectorChain * process.kt4CaloJetsL2L3 * process.kt4CaloL2L3Histos *
#process.kt6CaloL2L3CorrectorChain * process.kt6CaloJetsL2L3 * process.kt6CaloL2L3Histos *
#------ create the corrected pfjet collection and run the histogram module --------
process.ak4PFL2L3CorrectorChain * process.ak4PFJetsL2L3 * process.ak4PFL2L3Histos *
process.ak4PFCHSL2L3CorrectorChain * process.ak4PFCHSJetsL2L3 * process.ak4PFCHSL2L3Histos
#process.ak8PFL2L3CorrectorChain * process.ak8PFJetsL2L3 * process.ak8PFL2L3Histos *
#process.kt4PFL2L3CorrectorChain * process.kt4PFJetsL2L3 * process.kt4PFL2L3Histos *
#process.kt6PFL2L3CorrectorChain * process.kt6PFJetsL2L3 * process.kt6PFL2L3Histos *
#------ create the corrected jptjet collection and run the histogram module -------
#process.ak4JPTL2L3CorrectorChain * process.ak4JPTJetsL2L3 * process.ak4JPTL2L3Histos *
#------ create the corrected trackjet collection and run the histogram module -----
#process.ak4TrackL2L3CorrectorChain * process.ak4TrackJetsL2L3 * process.ak4TrackL2L3Histos
)








