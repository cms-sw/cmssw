import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

#-----------------------------
# DQM Environment & Specify inputs
#-----------------------------
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1)
)

#
#--- When read in DQM root file as input for certification
#process.source = cms.Source("EmptySource"
#)

#
#--- When read in RECO file including EDM from ME
process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string('RunsAndLumis'),
#   fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/h/hatake/scratch0/dqm/CMSSW_3_1_0_pre11_DQM/src/step2_DT2_1_RAW2DIGI_RECO_DQM_aaa.root')
#    fileNames = cms.untracked.vstring('file:reco_DQM_cruzet98154_v5.root',
#                                      'file:reco_DQM_cruzet98154_v7.root')
     fileNames = cms.untracked.vstring('file:reco_DQM_cruzet100945.root')
#    fileNames = cms.untracked.vstring('file:reco_DQM_cruzet100945_v1.root',
#                                      'file:reco_DQM_cruzet100945_v2.root')
#    fileNames = cms.untracked.vstring(
#    '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/FA72B935-0960-DE11-A902-000423D98DB4.root',
#    '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/0C547BAF-0C60-DE11-83C3-000423D98868.root')
#    fileNames = cms.untracked.vstring(
#    'file:/tmp/hatake/EADF3BE3-BE4F-DE11-8BB8-000423D9870C.root')
)

#-----

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

#-----------------------------
# Specify root file including reference histograms
#-----------------------------
#process.DQMStore.referenceFileName = '/afs/cern.ch/user/h/hatake/scratch0/dqm/CMSSW_3_1_0_pre11_DQM5/src/DQMOffline/JetMET/test/jetMETMonitoring_cruzet98154_v4.root'
#process.DQMStore.referenceFileName = '/afs/cern.ch/user/h/hatake/scratch0/dqm/CMSSW_3_1_0_pre11_DQM5/src/DQMOffline/JetMET/test/reference.root'
process.DQMStore.referenceFileName = 'jetMETMonitoring_cruzet100945.root'

#-----------------------------
# Locate a directory in DQMStore
#-----------------------------
process.dqmInfoJetMET = cms.EDAnalyzer("DQMEventInfo",
                subSystemFolder = cms.untracked.string('JetMET')
                )

#-----------------------------
# JetMET Certification Module 
#-----------------------------
process.load("DQMOffline.JetMET.dataCertificationJetMET_cff")
process.dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
#
#--- Always define reference root file by process.DQMStore.referenceFileName
                              refFileName    = cms.untracked.string(""),
#
#--- 0: harvest EDM files, 1: read in DQM root file
                              TestType       = cms.untracked.int32(0),
#
#--- When read in DQM root file as input for certification
#                             fileName       = cms.untracked.string("jetMETMonitoring_cruzet98154.root"),
#
#--- When read in RECO file including EDM from ME
                              fileName       = cms.untracked.string(""),
#
#--- Do note save here. Save output by dqmSaver
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string(""),
#
                              Verbose        = cms.untracked.int32(0)
)

#-----------------------------
# 
#-----------------------------
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *

#-----------------------------
# 
#-----------------------------
#process.p = cms.Path(process.dqmInfoJetMET*process.dataCertificationJetMET)

process.p = cms.Path(process.EDMtoME
                     * process.dqmInfoJetMET
                     * process.jetMETHLTOfflineClient
                     * process.dataCertificationJetMETSequence
                     * process.dqmSaver)

