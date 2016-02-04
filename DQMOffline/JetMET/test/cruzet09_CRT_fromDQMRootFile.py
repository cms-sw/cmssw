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
process.source = cms.Source("EmptySource"
)

#
#--- Load dqm root files
process.dqmFileReaderJetMET = cms.EDFilter("DQMFileReader",
  FileNames = cms.untracked.vstring('jetMETMonitoring_cruzet100945.root',
                                    'jetMETMonitoring_cruzet100945b.root')
)

#-----

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

#-----------------------------
# Specify root file including reference histograms
#-----------------------------
process.DQMStore.referenceFileName = 'jetMETMonitoring_cruzet100945.root'
#process.DQMStore.verbose = 5
#process.DQMStore.collateHistograms = True

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
#                             fileName       = cms.untracked.string("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/PromptReco/100/945/DQM_V0001_R000100945__Cosmics__Commissioning09-PromptReco-v3__RECO.root"),
#                             fileName       = cms.untracked.string("/uscms_data/d2/hatake/DQM-data/DQM_V0001_R000100945__Cosmics__Commissioning09-PromptReco-v3__RECO.root"),
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
process.p = cms.Path(process.dqmFileReaderJetMET
                     * process.dqmInfoJetMET
                     * process.dataCertificationJetMETSequence
                     * process.dqmSaver)


