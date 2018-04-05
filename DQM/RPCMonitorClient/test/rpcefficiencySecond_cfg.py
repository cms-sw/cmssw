import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

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
fileNames = cms.untracked.vstring('file:/tmp/cimmino/first.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.rpcEfficiencySecond = DQMEDHarvester("RPCEfficiencySecond",
SaveFile = cms.untracked.bool(True),
NameFile = cms.untracked.string('/tmp/cimmino/RPCEfficiency.root'),
debug = cms.untracked.bool(False),
barrel = cms.untracked.bool(True),
endcap = cms.untracked.bool(True)
)

process.p = cms.Path(process.EDMtoMEConverter*process.rpcEfficiencySecond)
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False

