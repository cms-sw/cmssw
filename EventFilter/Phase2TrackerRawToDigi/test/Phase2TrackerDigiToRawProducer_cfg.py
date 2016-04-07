import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DigiToRaw")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
        destinations  = cms.untracked.vstring('logtrace' ),
        logtrace      = cms.untracked.PSet( threshold  = cms.untracked.string('DEBUG') ),
        debugModules  = cms.untracked.vstring( 'Phase2TrackerDigiToRawProducer', 'Phase2TrackerFEDBuffer' )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Input source
process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring( 'root://xrootd.ba.infn.it/'+sys.argv[-1])
    fileNames = cms.untracked.vstring( 'file://'+sys.argv[-1])
)


process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('DummyCablingTxt_cfi')
process.load('EventFilter.Phase2TrackerRawToDigi.Phase2TrackerDigiToRawProducer_cfi')
process.Phase2TrackerDigiToRawProducer.ProductLabel = cms.InputTag("siPhase2Clusters")

# imported from runthematrix
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('digi2raw.root'),
    outputCommands = cms.untracked.vstring(
      # 'drop *',
      'keep *_Phase2TrackerDigiToRawProducer_*_*'
      )
    )


process.p = cms.Path(process.Phase2TrackerDigiToRawProducer)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023Muon 
#call to customisation function cust_2023Muon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023Muon(process)

process.e = cms.EndPath(process.out)

