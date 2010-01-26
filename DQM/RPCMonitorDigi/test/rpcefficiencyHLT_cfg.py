# The following comments couldn't be translated into the new config version:

#keep the logging output to a nice level

import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCSegmentEff")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"
process.prefer("GlobalTag")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000),
)

process.options = cms.untracked.PSet(
#   SkipEvent = cms.untracked.vstring(["ProductNotFound"])
)

process.source = cms.Source("PoolSource",
#   fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/Cosmics/RECO/v1/000/119/580/76BDF4A8-1ACA-DE11-9177-000423D98E6C.root')
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_5_0_pre2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0010/B2084415-22EE-DE11-9BA9-002618943842.root')
#   fileNames = cms.untracked.vstring('file:/tmp/carrillo/B2084415-22EE-DE11-9BA9-002618943842.root')
    fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/E68C41F8-0BE9-DE11-9C81-000423D94C68.root')

)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('/tmp/carrillo/RPCEfficiency.log')
)

process.museg = cms.EDFilter("RPCEfficiency",

    incldt = cms.untracked.bool(True),
    incldtMB4 = cms.untracked.bool(True),
    inclcsc = cms.untracked.bool(True),

    debug = cms.untracked.bool(True),
    inves = cms.untracked.bool(True),
    
    DuplicationCorrection = cms.untracked.int32(1),
	
    rangestrips = cms.untracked.double(1.),
    rangestripsRB4 = cms.untracked.double(4.),
    MinCosAng = cms.untracked.double(0.99),
    MaxD = cms.untracked.double(80.0),
    MaxDrb4 = cms.untracked.double(150.0),

 #   cscSegments = cms.untracked.string('cscSegments'),
 #   dt4DSegments = cms.untracked.string('dt4DSegments'),

    cscSegments = cms.untracked.string('hltCscSegments'),
    dt4DSegments = cms.untracked.string('hltDt4DSegments'),


    folderPath = cms.untracked.string('HLT/HLTMonMuon/RPC/'),

    EffSaveRootFile = cms.untracked.bool(True)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('/tmp/carrillo/output.root')
)

process.p = cms.Path(process.museg*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)

process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False


