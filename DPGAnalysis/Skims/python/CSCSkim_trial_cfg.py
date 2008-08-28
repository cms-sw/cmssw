import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
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
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/056/591/8E478B5A-3C6F-DD11-AC01-000423D985B0.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/289/1E1407F1-106D-DD11-97A7-000423D985E4.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/313/3C2E24C9-106D-DD11-9503-000423D6B358.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/381/16B04CC1-2A6D-DD11-8580-000423D6BA18.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/381/C2E66C73-2A6D-DD11-AA12-000423D6AF24.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/394/14AE2A35-2D6D-DD11-8EBB-000423D9853C.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/400/18DA955F-3F6D-DD11-B38C-0016177CA7A0.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/404/0074DB01-3F6D-DD11-BDC4-001617C3B6C6.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/404/1C48FCC8-3E6D-DD11-BBB6-000423D6AF24.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/404/28DBA2A1-3E6D-DD11-986E-000423D6CA02.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/404/2ACCFB81-3F6D-DD11-BDAF-001617DBD288.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/404/32E3B105-416D-DD11-80B6-000423D9880C.root'
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

process.mySkim = cms.Path(process.muonCSCDigis*process.csc2DRecHits*process.cscSegments*process.cscSkim)


#### output 
process.outputSkim = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("/tmp/schmittm/selectedEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RECO'),
      filterName = cms.untracked.string('CSCSkim_trial')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('mySkim'))
)

process.outpath = cms.EndPath(process.outputSkim)

#
