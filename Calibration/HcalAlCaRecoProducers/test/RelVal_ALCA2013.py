# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: RelVal -s ALCA:HcalCalPedestal --data --scenario=pp -n 100 --conditions auto:run2_data_FULL --datatier USER --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --customise=SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --magField 38T_PostLS1 --processName=USER --filein file:RelVal_RECO.root --fileout file:RelVal_ALCA.root --no_exec
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
    input = cms.untracked.int32(100000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/043606C2-8745-E311-B533-0025901D5D86.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0A25D47F-AF47-E311-8D72-002481E734DA.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0C01FFCD-3E48-E311-9F3E-002481E0D7DA.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0CCE30F7-F045-E311-9484-003048F02D4A.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0CE91666-9446-E311-A79A-003048D49A3E.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0E65C8EF-8F46-E311-8253-003048F1D938.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/0EFB341D-F047-E311-A6C5-003048D37524.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/10096C40-C846-E311-8646-C860001BD8A2.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/12BABF91-8D47-E311-B9B1-003048F16F3A.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/12D70926-F147-E311-8EB4-003048F1B916.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/168DC999-5447-E311-BBDA-003048D2BC5E.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/1AD24E33-F147-E311-A0D9-0025B32038DC.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/1C3FF13F-1047-E311-B1B9-0025901AF676.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/1C45F9CF-3745-E311-A314-0025901D6260.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/1EBE397E-AF45-E311-8C9E-003048F0E512.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/2257BB5D-3C47-E311-B320-003048F0E5BE.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/22E08E08-4848-E311-95B2-003048D374D2.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/2682F179-CA46-E311-8A37-00237DE0C490.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/284DE1EA-FC45-E311-A282-003048F0E7A4.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/28598EE7-DF48-E311-AF8B-5404A63886AB.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/2C00A7C9-2648-E311-B593-003048F1DB64.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/2C0345F7-7A48-E311-9521-5404A6388692.root',
'/store/data/Commissioning2013/TestEnablesEcalHcalDT/RAW/v1/00000/2CB5C3F0-3E46-E311-9850-003048F1DE4A.root'
   ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('RelVal nevts:100'),
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
        'keep *_gtDigisAlCaMB_*_*', 
        'keep HBHERecHitsSorted_hbherecoMB_*_*', 
        'keep HORecHitsSorted_horecoMB_*_*', 
        'keep HFRecHitsSorted_hfrecoMB_*_*', 
        'keep HFRecHitsSorted_hfrecoMBspecial_*_*', 
        'keep HBHERecHitsSorted_hbherecoNoise_*_*', 
        'keep HORecHitsSorted_horecoNoise_*_*', 
        'keep HFRecHitsSorted_hfrecoNoise_*_*')
)

# Other statements
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_FULL', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalPedestalOutPath = cms.EndPath(process.ALCARECOStreamHcalCalPedestal)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalPedestal,process.endjob_step,process.ALCARECOStreamHcalCalPedestalOutPath)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.CustomConfigs
from HLTrigger.Configuration.CustomConfigs import L1THLT 

#call to customisation function L1THLT imported from HLTrigger.Configuration.CustomConfigs
process = L1THLT(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions

