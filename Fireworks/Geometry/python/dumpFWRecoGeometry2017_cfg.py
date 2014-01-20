import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load('Configuration.Geometry.GeometryExtended2017_cff')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2017', '')

process.add_(cms.ESProducer("FWRecoGeometryESProducer"))

#Adding Timing service:
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
       wantSummary = cms.untracked.bool(True)
       )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(1)
       )
process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                             level = cms.untracked.int32(1)
                             )

process.p = cms.Path(process.dump)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2017 

#call to customisation function cust_2017 imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2017(process)
