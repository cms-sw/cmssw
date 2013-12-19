# Auto generated configuration file
# using:
# Revision: 1.1
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: test_11_a_1 -s RAW2DIGI,RECO,DQM -n 500 --eventcontent DQM --conditions auto:com10 --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --data --customise DQMTools/Tests/customDQM.py --no_exec --python_filename=test_11_a_1.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.RawToDigi.remove(process.csctfDigis)
process.RawToDigi.remove(process.dttfDigis)
process.RawToDigi.remove(process.gctDigis)
#process.RawToDigi.remove(process.gtDigis)
process.RawToDigi.remove(process.gtEvmDigis)
process.RawToDigi.remove(process.siPixelDigis)
process.RawToDigi.remove(process.siStripDigis)
process.RawToDigi.remove(process.ecalDigis)
process.RawToDigi.remove(process.ecalPreshowerDigis)
process.RawToDigi.remove(process.hcalDigis)
process.RawToDigi.remove(process.muonCSCDigis)
process.RawToDigi.remove(process.muonDTDigis)
process.RawToDigi.remove(process.muonRPCDigis)
process.RawToDigi.remove(process.castorDigis)
#process.RawToDigi.remove(process.scalersRawToDigi)

process.DQMOffline.remove(process.DQMOfflinePreDPG)

# Removing other DQM modules form the DQMOfflinePrePOG
process.DQMOfflinePrePOG.remove(process.muonMonitors)
process.DQMOfflinePrePOG.remove(process.jetMETDQMOfflineSource)
process.DQMOfflinePrePOG.remove(process.egammaDQMOffline)
#process.DQMOfflinePrePOG.Remove(process.l1TriggerDqmOffline)
process.DQMOfflinePrePOG.remove(process.triggerOfflineDQMSource)
process.DQMOfflinePrePOG.remove(process.pvMonitor)
process.DQMOfflinePrePOG.remove(process.prebTagSequence)
process.DQMOfflinePrePOG.remove(process.bTagPlotsDATA)
process.DQMOfflinePrePOG.remove(process.alcaBeamMonitor)
process.DQMOfflinePrePOG.remove(process.dqmPhysics)
process.DQMOfflinePrePOG.remove(process.produceDenoms)
process.DQMOfflinePrePOG.remove(process.pfTauRunDQMValidation)

process.l1TriggerDqmOffline.remove(process.l1TriggerOffline)
process.l1TriggerDqmOffline.remove(process.l1tSync_Offline)
process.l1TriggerDqmOffline.remove(process.l1tEfficiencyMuons_offline)
process.l1TriggerDqmOffline.remove(process.l1TriggerEmulatorOffline)

# Removing other L1T DQM modules 
#process.l1tMonitorOnline.remove(process.bxTiming)
#process.l1tMonitorOnline.remove(process.l1tDttf)
#process.l1tMonitorOnline.remove(process.l1tCsctf) 
#process.l1tMonitorOnline.remove(process.l1tRpctf)
#process.l1tMonitorOnline.remove(process.l1tGmt)
#process.l1tMonitorOnline.remove(process.l1tGt)
#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)
#process.l1tMonitorOnline.remove(process.l1tBPTX)
#process.l1tMonitorOnline.remove(process.l1tRate)
#process.l1tMonitorOnline.remove(process.l1tRctSeq)
#process.l1tMonitorOnline.remove(process.l1tGctSeq)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('test_11_a_1 nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('L1TOffline_L1TriggerOnly_job1_RAW2DIGI_RECO_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
#process.reconstruction_step = cms.Path(process.reconstruction)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.DQMoutput_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.dqmoffline_step,process.endjob_step,process.DQMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from DQMTools.Tests.customDQM
from DQMTools.Tests.customDQM import customise

#call to customisation function customise imported from DQMTools.Tests.customDQM
process = customise(process)

# End of customisation functions
