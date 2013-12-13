################################################################################
#
# run_ak4L2L3_cfg.py
# ------------------
#
# This configuration demonstrates how to run the L2L3 correction producers
# for AntiKt R=0.5 CaloJets and PFJets.
#
# Note that you can switch between different sets of corrections
# (e.g. Summer09 and Summer09_7TeV) by simply setting the below 'era'
# parameter accordingly!
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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/CEAAC1F4-0BB7-DE11-93D7-000423D99AA2.root')
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
JetCorrectionEra.era = 'Summer09_7TeV'
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')


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

process.ak4CaloHistos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak4CaloJets'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak4CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak4CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak4PFHistos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak4PFJets'),
    histograms = cms.VPSet(jetPtHistogram)
    )
process.ak4PFL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak4PFJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )

#
# RUN!
#
process.run = cms.Path(
    process.ak4CaloJetsL2L3*process.ak4CaloHistos*process.ak4CaloL2L3Histos*
    process.ak4PFJetsL2L3*  process.ak4PFHistos*  process.ak4PFL2L3Histos
    )
