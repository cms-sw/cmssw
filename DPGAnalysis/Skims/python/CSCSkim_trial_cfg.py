import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

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
  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root'),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
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
process.GlobalTag.globaltag = 'CRAFT_ALL_V9::All' 
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
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string("cscskimEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_trial')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('mySkim'))
)

process.outpath = cms.EndPath(process.outputSkim)

