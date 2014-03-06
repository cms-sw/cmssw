import FWCore.ParameterSet.Config as cms

process = cms.Process("ME0SegmentMatching")

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }

)

process.me0SegmentProducer = cms.EDProducer("ME0SegmentProducer")

process.me0SegmentMatcher = cms.EDProducer("ME0SegmentMatcher",
                                           DebugHistos = cms.string('DebugHistos.root'),
                                           debug = cms.bool(True)
)

process.me0MuonConverter = cms.EDProducer("ME0MuonConverter")

process.p = cms.Path(process.me0SegmentProducer+
                     process.me0SegmentMatcher+
                     process.me0MuonConverter)
process.PoolSource.fileNames = [
    #'file:/afs/cern.ch/work/d/dnash/PixelProduction/TryFour/CMSSW_6_1_2_SLHC8/src/FastSimulation/Configuration/test/20GeV.root'
    #'file:/afs/cern.ch/work/d/dnash/PixelProduction/ReCheckout/CMSSW_6_1_2_SLHC8/src/FastSimulation/Configuration/test/Zmumu.root'
    #'file:/tmp/dnash/Zmumu.root'
    #'file:/tmp/dnash/Zmumu_620.root'
    #'file:/afs/cern.ch/work/d/dnash/ME0Segments/FullSimPixel/TryNoME0SLHC8/CMSSW_6_2_0_SLHC8/src/FastSimulation/Configuration/test/MyFirstFamosFile_2.root'
    'file:/tmp/dnash/MyFirstFamosFile_2.root'
]


process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
#                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('FirstTest.root')
)

process.outpath = cms.EndPath(process.o1)
