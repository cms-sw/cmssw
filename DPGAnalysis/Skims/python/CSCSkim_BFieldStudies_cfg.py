import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIMb")


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.6 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/CSCSkim_trial_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT CSCSkim skim B')
)

#
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring (
'/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V4_CSCSkim_trial_CSCSkim_trial_v3/0268/1EA84E82-68EE-DD11-88FF-0019B9E7CEEF.root',
                                         )
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

#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
process.load("DPGAnalysis/Skims/CSCSkim_cfi")
process.cscSkim.typeOfSkim = cms.untracked.int32(9)

#### the path

process.BfieldStudySkim = cms.Path(process.cscSkim)


#### output 
process.outputSkim = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string("CSCEvents_BFieldStudy.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_BFieldStudies')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('BfieldStudySkim'))
)

process.outpath = cms.EndPath(process.outputSkim)
#
