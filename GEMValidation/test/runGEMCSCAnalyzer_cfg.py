import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMCSCANA")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

# the analyzer configuration
process.load('GEMCode.GEMValidation.GEMCSCAnalyzer_cfi')
#process.GEMCSCAnalyzer.verbose = 2
#process.GEMCSCAnalyzer.ntupleTrackChamberDelta = False
#process.GEMCSCAnalyzer.ntupleTrackEff = True
process.GEMCSCAnalyzer.ntupleTrackChamberDelta = True
process.GEMCSCAnalyzer.ntupleTrackEff = True
process.GEMCSCAnalyzer.minPt = 1.5
#process.GEMCSCAnalyzer.simTrackMatching.verboseSimHit = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseGEMDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCStub = 1
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyGEM = False
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsGEM = False
process.GEMCSCAnalyzer.simTrackMatching.gemRecHit.input = ""
process.GEMCSCAnalyzer.simTrackMatching.minNHitsChamber = 3

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_l1.root'
    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('gem-csc_stub_ana.root')
)

process.p = cms.Path(process.GEMCSCAnalyzer)

