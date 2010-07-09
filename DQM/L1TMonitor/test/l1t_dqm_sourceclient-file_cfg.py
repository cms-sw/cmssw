#
# cfg file to run online L1 Trigger DQM
#
# V M Ghete 2010-07-09


import FWCore.ParameterSet.Config as cms
import sys

# choose the environment you run
#l1DqmEnv = 'live'
#l1DqmEnv = 'playback'
l1DqmEnv = 'file'

process = cms.Process("DQM")

# check that a valid choice for environment exists

if not ((l1DqmEnv == 'live') or l1DqmEnv == 'playback' or l1DqmEnv == 'file') : 
    print 'No valid input source was chosen. Your value for l1DqmEnv input parameter is:'  
    print 'l1DqmEnv = ', l1DqmEnv
    print 'Available options: "live", "playback", "file" '
    sys.exit()

#----------------------------
# Event Source
#

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.inputsource_cfi")
    process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring("*")
            )
    process.EventStreamHttpReader.consumerName = 'L1T DQM Consumer'
    process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(25.0)
 
elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
else : 
    # running on a file
    process.load("DQM.L1TMonitor.inputsource_file_cfi")
    
      
#----------------------------
# DQM Environment
#

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'L1T'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

    #
    # load and configure modules via Global Tag
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')

elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
else : 
    # running on a file
    process.load("DQM.L1TMonitor.environment_file_cfi")

    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    process.GlobalTag.globaltag = 'GR_R_36X_V10::All'
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

#-----------------------------
#
#  L1 DQM SOURCES
#

process.load("DQM.L1TMonitor.L1TMonitor_cff")
process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")

process.load("DQM.TrigXMonitor.L1Scalers_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")
process.l1s.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.l1s.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM") 
process.l1tsClient.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM")
process.p3 = cms.EndPath(process.l1s+process.l1tsClient)

process.load("DQM.TrigXMonitor.HLTScalers_cfi")
process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")
process.hlts.dqmFolder = cms.untracked.string("L1T/HLTScalers_SM")
process.hltsClient.dqmFolder = cms.untracked.string("L1T/HLTScalers_SM")
process.p = cms.EndPath(process.hlts+process.hltsClient)

# removed modules
#process.hltMonScal.remove("l1tscalers")

##  Available data masks (case insensitive):
##    all, gt, muons, jets, taujets, isoem, nonisoem, met
process.l1tEventInfoClient.dataMaskedSystems = cms.untracked.vstring(
    "Muons","Jets","TauJets","IsoEm","NonIsoEm","MET"
    )

##  Available emulator masks (case insensitive):
##    all, dttf, dttpg, csctf, csctpg, rpc, gmt, ecal, hcal, rct, gct, glt
process.l1tEventInfoClient.emulatorMaskedSystems = cms.untracked.vstring("All")



