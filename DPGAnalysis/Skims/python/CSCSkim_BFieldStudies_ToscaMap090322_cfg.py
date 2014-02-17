import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/CSCSkim_BFieldStudies_ToscaMap090322_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT CSCSkim skim B')
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0000/FE32B1E4-C7FA-DD11-A2FD-001A92971ADC.root'),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/708C5612-CFA5-DD11-AD52-0019DB29C5FC.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/38419E41-D1A5-DD11-8B68-001617C3B6E2.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/2CDF3B0F-CFA5-DD11-AE18-000423D99A8E.root')
 )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load("Configuration/StandardSequences/Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_UpdatedMap_cff")
# trick to make it work with newnew magfield (not in 229)
process.VolumeBasedMagneticFieldESProducer.version='grid_1103l_090322_3_8t'

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
