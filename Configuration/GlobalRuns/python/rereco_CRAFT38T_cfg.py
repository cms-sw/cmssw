
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec2")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
#process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/ReconstructionCosmics_cff')
process.load('DQMOffline/Configuration/DQMOfflineCosmics_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContentCosmics_cff')
# Magnetic field: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
# Conditions (Global Tag is used here):
process.GlobalTag.globaltag = "CRAFT0831X_V1::All"

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(-1))

#Drop old reco
process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(523),
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0012/1A8C28D2-0402-DE11-84C0-0018F3D09644.root'
#    '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0006/20993712-7C00-DE11-8BAA-003048678B5E.root',
#    '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0007/7AC674EA-7F00-DE11-A0AB-0018F3D09706.root',
#    '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0008/5EF7C8EC-9B00-DE11-B1C5-00304867920C.root'
    ),
    inputCommands = cms.untracked.vstring('drop *')
)
process.source.inputCommands.extend(process.RAWEventContent.outputCommands)
process.source.inputCommands.append('drop *_*_*_Rec')
process.source.inputCommands.append('keep *_eventAuxiliaryHistoryProducer_*_*')
process.source.dropDescendantsOfDroppedBranches=cms.untracked.bool(False)


# output module
#

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO')),
    fileName = cms.untracked.string('/tmp/malgeri/promptRerecoCosmics.root')
)

process.FEVT.outputCommands.append('keep *_eventAuxiliaryHistoryProducer_*_*')


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GlobalRuns/python/rereco_CRAFT38T_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT ReReco with Mag field at 3.8T')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false

# Path and EndPath definitions

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstructionCosmics)
process.dqmoffline_step = cms.Path(process.DQMOfflineCosmics)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.FEVT)

# remove crashing modules
process.dqmoffline_step.remove(process.hcalOfflineDQMSource)
# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.out_step)


