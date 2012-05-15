import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuGolden")

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuGolden_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1

#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('')
#process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/1CD6D0A6-1E64-DF11-BB60-001D09FD0D10.root',
   # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0007/CAE2081C-48B5-DE11-9161-001D09F29321.root',
    )
)


process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuGolden.root")
)

zPlots = cms.PSet(
    histograms = cms.VPSet(
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zMass"),
    description = cms.untracked.string("Z mass [GeV/c^{2}]"),
    plotquantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu1Pt"),
    description = cms.untracked.string("Highest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("max(daughter(0).pt,daughter(1).pt)")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu2Pt"),
    description = cms.untracked.string("Lowest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)")
    )
    )
)




process.goodZToMuMuPlots = cms.EDFilter(
    "CandViewHistoAnalyzer",
    zPlots,
#    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    src = cms.InputTag("zmmCands"),
    filter = cms.bool(False)
)




process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.ewkZMuMuGoldenPath = cms.Path(
    process.ewkZMuMuGoldenSequence *
    process.goodZToMuMuPlots
)



process.endPath = cms.EndPath( 
   process.eventInfo 
)
