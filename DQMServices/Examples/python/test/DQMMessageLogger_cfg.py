
import FWCore.ParameterSet.Config as cms


process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''
process.load("FWCore.Modules.logErrorHarvester_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("Configuration/StandardSequences/RawToDigi_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")

process.GlobalTag.globaltag = "STARTUP3XY_V9::All"
process.prefer("GlobalTag")
#process.Tracer = cms.Service("Tracer")

process.load("DQMServices.Components.DQMMessageLogger_cfi")
process.load("DQMServices.Components.DQMMessageLoggerClient_cfi")

process.dqmSaver.workflow = cms.untracked.string('/workflow/for/mytest')



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)


process.source = cms.Source("PoolSource",                          
                            fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3XY_V9-v1/0003/C45E3207-9CBD-DE11-ADC3-001731A28543.root"

    )
                            
)



process.p = cms.Path(process.RawToDigi*process.reconstruction*process.logErrorHarvester*process.DQMMessageLogger*process.DQMMessageLoggerClient + process.dqmSaver)

