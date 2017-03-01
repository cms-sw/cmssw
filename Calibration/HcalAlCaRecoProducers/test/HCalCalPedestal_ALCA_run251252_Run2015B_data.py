# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: HCalCalPedestal -s ALCA:HcalCalPedestal --data --scenario=pp --conditions auto:run2_data_FULL -n -1 --customise=SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --magField AutoFromDBCurrent --processName=USER --filein root://eoscms//eos/cms/store/data/Run2015B/TestEnablesEcalHcal/RAW/v1/000/251/252/00000/82A5CE77-D025-E511-8EC3-02163E0136E2.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('USER')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/data/Run2015B/TestEnablesEcalHcal/RAW/v1/000/251/252/00000/82A5CE77-D025-E511-8EC3-02163E0136E2.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('HCalCalPedestal nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition


# Additional output definition
process.ALCARECOStreamHcalCalPedestal = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalPedestal')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('ALCARECOHcalCalPedestal')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('ALCARECOHcalCalPedestal.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_gtDigisAlCaPedestal_*_*', 
        'keep HBHERecHitsSorted_hbherecoPedestal_*_*', 
        'keep HORecHitsSorted_horecoPedestal_*_*', 
        'keep HFRecHitsSorted_hfrecoPedestal_*_*')
)

# Other statements
process.ALCARECOEventContent.outputCommands.extend(process.OutALCARECOHcalCalPedestal_noDrop.outputCommands)
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_FULL', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalPedestalOutPath = cms.EndPath(process.ALCARECOStreamHcalCalPedestal)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalPedestal,process.endjob_step,process.ALCARECOStreamHcalCalPedestalOutPath)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions

