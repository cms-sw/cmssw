# Auto generated configuration file
# using: 
# Revision: 1.285 
# Source: /cvs/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: reco -s RAW2DIGI,RECO --eventcontent RECO --conditions auto:startup -n 100 --data --dbs find file,file.parent where dataset=/RelValZEE/CMSSW_3_11_0_pre5-START311_V0-v1/GEN-SIM-RECO --process=RERECO --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RERECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
    'file:/build/argiro/Run2010B_EGMonitor_RAW_v1_000_149_509_a.root'
    )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('reco nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('reco_RAW2DIGI_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_R_43_V2::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('testFlags.root')
                                   )

process.severityAnalyzer=cms.EDAnalyzer('EcalSeverityLevelAlgoAnalyzer')
process.testp = cms.Path(process.severityAnalyzer)


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.testp,process.endjob_step,process.RECOoutput_step)




