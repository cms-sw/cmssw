import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }

)

process.Test = cms.EDAnalyzer("TestAnalyzer_Final",
                              HistoFile = cms.string('OutputTestHistos_TestLocal.root'),
                              
)

process.p = cms.Path(process.Test)
process.PoolSource.fileNames = [
    #'file:FirstTest.root'
    #'file:/afs/cern.ch/work/d/dnash/ME0Segments/CommitToCMSSW/CMSSW_6_1_2_SLHC8/src/RecoMuon/MuonIdentification/test/FirstTest.root'
    #'file:/tmp/dnash/Zmumu_FlatMuonPt_SLHC8.root'
    #'file:/afs/cern.ch/work/d/dnash/ME0Segments/FullSimPixel/CMSSW_6_2_0_SLHC8/src/'
    #'file:/afs/cern.ch/work/d/dnash/ME0Segments/FullSimPixel/CMSSW_6_2_0_SLHC8/src/ZMMTest.root'
    'file:/tmp/dnash/ZMMTest_Again.root'

]
