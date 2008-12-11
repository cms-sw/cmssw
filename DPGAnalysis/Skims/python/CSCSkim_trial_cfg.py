import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/CSCSkim_trial_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT CSCSkim skim')
)

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
           '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/D424EBA5-55A0-DD11-A8BF-000423D9853C.root',
                  '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/C67EDF0D-49A0-DD11-9403-001617DBD332.root'),
                           secondaryFileNames = cms.untracked.vstring(
           '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/122/6E2601EC-3FA0-DD11-BA50-000423D986A8.root',
                  '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/122/C240B0B2-47A0-DD11-A6AD-001617C3B654.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_V4P::All' 
process.prefer("GlobalTag")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")
process.es_prefer_MagneticField = cms.ESPrefer("VolumeBasedMagneticFieldESProducer")

#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
process.load("DPGAnalysis/Skims/CSCSkim_cfi")

#### the path

process.mySkim = cms.Path(process.cscSkim)


#### output 
process.outputSkim = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string("cscskimEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_trial')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('mySkim'))
)

process.outpath = cms.EndPath(process.outputSkim)

#
