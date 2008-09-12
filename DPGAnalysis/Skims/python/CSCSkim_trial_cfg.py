import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/CSCSkim_trial_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 CSCSkim skim')
)

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring (
      '/store/data/BeamCommissioning08/BeamHalo/RECO/v1/000/062/232/04FF0C7D-7280-DD11-8FCD-000423D98F98.root'
       )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRUZET4_V2::All' 
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
    fileName = cms.untracked.string("/tmp/arizzi/selectedEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RECO'),
      filterName = cms.untracked.string('CSCSkim_trial')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('mySkim'))
)

process.outpath = cms.EndPath(process.outputSkim)

#
