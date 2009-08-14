
import FWCore.ParameterSet.Config as cms

process = cms.Process("DTLPtest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")


# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EC608867-AE6B-DE11-9952-000423D94E1C.root',
      '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EAC844CB-E16B-DE11-87C6-001D09F29533.root',
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/9C372501-B06B-DE11-81BC-001D09F24448.root',
     '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/9887DA27-B06B-DE11-B743-000423D94E1C.root',
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6AE5271F-AF6B-DE11-A616-001D09F232B9.root',
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/38BCDD1E-AF6B-DE11-A968-001D09F2AD4D.root',
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/2011B6F1-B26B-DE11-82CC-000423D6CAF2.root',
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/10C7AFB7-AC6B-DE11-8C78-001D09F28E80.root'

        ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = "MC_31X_V3::All"

process.load("RecoLocalMuon.DTRecHit.dt1DRecHits_ParamDrift_cfi")
process.load("RecoLocalMuon.DTSegment.dt2DSegments_LPPatternReco2D_ParamDrift_cfi")
#process.load("RecoLocalMuon.DTSegment.dt2DSegments_CombPatternReco2D_ParamDrift_cfi")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")


dtlocalreco = cms.Sequence(process.dt1DRecHits*process.dt2DSegments)

process.jobPath = cms.Path(process.muonDTDigis + dtlocalreco)


