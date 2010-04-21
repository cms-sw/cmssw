################################################################################
#
# run_ak5L2L3_cfg.py
# ------------------
#
# This configuration demonstrates how to run the L2L3 correction producers
# for kT R=0.4 CaloJets and PFJets.
#
# Note that you can switch between different sets of corrections
# (e.g. Summer09 and Summer09_7TeV) by simply setting the below 'era'
# parameter accordingly!
#
# The services and producers necessary to provide L2 & L3 corrections for
# kT R=0.4 jets are cloned from the already existing versions for
# AntiKt R=0.5. Any other available algorithms (e.g. SISCone) can be
# configured accordingly.
#
# The job creates 'histos.root' which contains the jet pT spectra for
# calorimeter and pflow jets before and after the application of the
# L2 (relative) and L3 (absolute) corrections.
#
################################################################################

import FWCore.ParameterSet.Config as cms

#!
#! PROCESS
#!
process = cms.Process("AK5L2L3")


#!
#! INPUT
#!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0014/82C13C57-FD49-DF11-B238-003048678DA2.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0014/66385781-E549-DF11-B8A6-00261894383E.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/DAF9E8AD-9749-DF11-ABDD-003048678FB2.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/92AB1865-9749-DF11-8AC9-00261894383E.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/5828E251-AD49-DF11-9C11-0018F3D096BC.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/287CF481-9749-DF11-9514-003048679162.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/241614BA-9A49-DF11-9B42-0018F3D0965A.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/164023A0-8E49-DF11-B1D5-0018F3D096CE.root',
        '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/MC_36Y_V4-v1/0013/06FBE314-9A49-DF11-ADD2-00261894393A.root'
        )
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
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionCondDB_cff')



#!
#! MAKE SOME HISTOGRAMS
#!
jetPtHistogram = cms.PSet(min          = cms.untracked.double(     10),
                          max          = cms.untracked.double(    200),
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
    src = cms.InputTag('ak5CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5PFHistos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5PFJets'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak5PFL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5PFJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )

#
# RUN!
#
process.run = cms.Path(
    process.ak5CaloJetsL2L3*process.ak5CaloHistos*process.ak5CaloL2L3Histos*
    process.ak5PFJetsL2L3*  process.ak5PFHistos*  process.ak5PFL2L3Histos
    )
