# The following comments couldn't be translated into the new config version:

#keep the logging output to a nice level

import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring('file:/tmp/carrillo/output.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.second = cms.EDAnalyzer("RPCEfficiencySecond",
    debug = cms.untracked.bool(False),
    barrel = cms.untracked.bool(True),
    endcap = cms.untracked.bool(True),
    SaveFile = cms.untracked.bool(False),
    folderPath = cms.untracked.string('HLT/HLTMonMuon/RPC/')
)

process.p = cms.Path(process.EDMtoMEConverter*process.second)
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False


