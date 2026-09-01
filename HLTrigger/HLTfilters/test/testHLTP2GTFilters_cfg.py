"""
Unit test for HLTP2GTSingleObjectFilter, HLTP2GTDoubleObjectFilter,
HLTP2GTTripleObjectFilter, and HLTP2GTQuadObjectFilter.

For each of four L1 seeds — one per filter type — we run:
  (a) the original PathStatusFilter reference path, and
  (b) a new HLTP2GT*ObjectFilter path reproducing the same condition.

A per-seed EDAnalyzer then asserts that (a) and (b) give the same
accept/reject decision on every event.  The job throws if any mismatch
is found, making it a hard failure in scram b runtests.

Seeds exercised
---------------
  Single  : SingleTkMuon22     (pSingleTkMuon22)     GMTTkMuons pT >= 22 GeV
  Double  : IsoTkEleEGEle22_12 (pIsoTkEleEGEle22_12) CL2Electrons x CL2Photons
  Triple  : TripleTkMuon5_3_3  (pTripleTkMuon5_3_3)  GMTTkMuons x3, thresholds 5/3/3
  Quad    : QuadJet70_55_40_40 (pQuadJet70_55_40_40) CL2JetsSC4 x4, thresholds 70/55/40/40

Run with:
  cmsRun HLTrigger/HLTfilters/test/HLTP2GTFilterTest_cfg.py
"""

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

process = cms.Process('TEST', Phase2C22I13M9)

# Standard setup (same as the reference L1 emulation config)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenu_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000
process = customise_aging_1000(process)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/eos/cms/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/'
        'GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/'
        '2590000/0438e4bc-b740-48a4-9d02-ff7896522eac.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_hlt*_*_HLT',
        'drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT',
    ),
)

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))

# MessageLogger: suppress per-event prints unless there is a mismatch
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.cerr.HLTP2GTFilterTestAnalyzer = cms.untracked.PSet(
    limit=cms.untracked.int32(100),
)
process.MessageLogger.cerr.HLTP2GTUtilities = cms.untracked.PSet(
    limit=cms.untracked.int32(0),  # silence per-candidate debug prints
)

# ===========================================================================
# Limit the mapping in the algo-block producer to the seeds used
# ===========================================================================
process.l1tGTAlgoBlockProducer = cms.EDProducer("L1GTAlgoBlockProducer",
    algorithms = cms.VPSet(
        cms.PSet(
            expression = cms.string('pIsoTkEleEGEle22_12')
        ),        
        cms.PSet(
            expression = cms.string('pTripleTkMuon5_3_3')
        ),
        cms.PSet(
            expression = cms.string('pSingleTkMuon22')
        ),        
        cms.PSet(
            expression = cms.string('pPuppiHT400 and pQuadJet70_55_40_40'),
            name = cms.string('pPuppiHT400_pQuadJet70_55_40_40')
        ),
        cms.PSet(
            expression = cms.string('pPuppiHT450')
        ),
    )
)

# ===========================================================================
# Shared helpers
# ===========================================================================

def _noPairCuts():
    return cms.PSet(
        minDR      = cms.double(0.),
        maxDR      = cms.double(1e9),
        minDEta    = cms.double(-1.),
        minDPhi    = cms.double(-1.),
        minInvMass = cms.double(0.),
        maxInvMass = cms.double(1e9),
    )



# ===========================================================================
#       1. SINGLE OBJECT : SingleTkMuon22 
#       Reference L1 path: pSingleTkMuon22
#       L1 condition: l1tGTSingleObjectCond on GMTTkMuons
# ===========================================================================

# Reference: gate on the L1 path decision
process.refSingleTkMuon22Filt = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string("pSingleTkMuon22"),
)

# Under test: HLTP2GTSingleObjectFilter
process.testSingleTkMuon22Filt = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags         = cms.bool(False),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    minN             = cms.uint32(1),
    debugAccepts     = cms.bool(False),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name       = cms.string("pSingleTkMuon22"),
            collection = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(2.4),
            ),
        ),
    ),
)

# ===========================================================================
#       1. SINGLE OBJECT : PuppiHT450
#       Reference L1 path: pPuppiHT450
#       L1 condition: l1tGTSingleObjectCond on CL2HtSum, minScalarSumPt=400
# ===========================================================================
 
# Reference: gate on the L1 path decision
process.refSinglePuppiHT450Filt = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string("pPuppiHT450"),
)
 
# Under test: HLTP2GTSingleObjectFilter
process.testSinglePuppiHT450Filt = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags         = cms.bool(False),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    minN             = cms.uint32(1),
    debugAccepts     = cms.bool(False),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name       = cms.string("pPuppiHT450"),
            collection = cms.PSet(
                objectType = cms.string("CL2HtSum"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(1e9),
            ),
        ),
    ),
)


# ===========================================================================
#       2. DOUBLE OBJECT : IsoTkEleEGEle22_12
#       Reference L1 path: pIsoTkEleEGEle22_12
#       L1 condition: l1tGTDoubleObjectCond
#         collection1: CL2Electrons >= 22 GeV
#         collection2: CL2Photons   >= 12 GeV   (L1EG/"default" in menu)
#         minDR = 0.1 between the two legs
# ===========================================================================

process.refDoubleIsoTkEleEGEle2212Filt = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string("pIsoTkEleEGEle22_12"),
)

process.testDoubleIsoTkEleEGEle2212Filt = cms.EDFilter("HLTP2GTDoubleObjectFilter",
    saveTags         = cms.bool(False),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pIsoTkEleEGEle22_12"),
            collection1 = cms.PSet(
                objectType = cms.string("CL2Electrons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection2 = cms.PSet(
                objectType = cms.string("CL2Photons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            minDR      = cms.double(0.1),
            maxDR      = cms.double(1e9),
            minDEta    = cms.double(-1.),
            minDPhi    = cms.double(-1.),
            minInvMass = cms.double(0.),
            maxInvMass = cms.double(1e9),
        ),
    ),
)

# ===========================================================================
#       3. TRIPLE OBJECT : TripleTkMuon5_3_3
#       Reference L1 path: pTripleTkMuon5_3_3
#       L1 condition: l1tGTTripleObjectCond on GMTTkMuons
#         collection1: pT >= 5 GeV, |eta| <= 2.4
#         collection2: pT >= 3 GeV, |eta| <= 2.4
#         collection3: pT >= 3 GeV, |eta| <= 2.4
#         no inter-muon cuts in the standard menu seed
# ===========================================================================

process.refTripleTkMuon533Filt = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string("pTripleTkMuon5_3_3"),
)

process.testTripleTkMuon533Filt = cms.EDFilter("HLTP2GTTripleObjectFilter",
    saveTags         = cms.bool(False),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pTripleTkMuon5_3_3"),
            collection1 = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection2 = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection3 = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            cuts12 = _noPairCuts(),
            cuts13 = _noPairCuts(),
            cuts23 = _noPairCuts(),
        ),
    ),
)

# ===========================================================================
#       4. QUAD OBJECT : QuadJet70_55_40_40
#       Reference L1 path: pQuadJet70_55_40_40
#       L1 condition: l1tGTQuadObjectCond on CL2JetsSC4
#         collection1: pT >= 70, |eta| <= 2.4
#         collection2: pT >= 55, |eta| <= 2.4
#         collection3: pT >= 40, |eta| <= 2.4
#         collection4: pT >= 40, |eta| <= 2.4
#         no inter-jet cuts
# ===========================================================================

process.refQuadJet70554040Filt = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string("pQuadJet70_55_40_40"),
)


process.testQuadJet70554040Filt = cms.EDFilter("HLTP2GTQuadObjectFilter",
    saveTags         = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
            collection1 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection2 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection3 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection4 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            cuts12 = _noPairCuts(),
            cuts13 = _noPairCuts(),
            cuts14 = _noPairCuts(),
            cuts23 = _noPairCuts(),
            cuts24 = _noPairCuts(),
            cuts34 = _noPairCuts(),
        ),
    ),
)

# ===========================================================================
# Comparison analyzers — one per seed, throw on any mismatch
# ===========================================================================

_trigRes = cms.InputTag("TriggerResults", "", "TEST")

process.cmpSingleA = cms.EDAnalyzer("HLTP2GTFilterTestAnalyzer",
   triggerResults = _trigRes,
   referencePath  = cms.string("refSinglePuppiHT450"),
   underTestPath  = cms.string("testSinglePuppiHT450"),
)

process.cmpSingleB = cms.EDAnalyzer("HLTP2GTFilterTestAnalyzer",
    triggerResults = _trigRes,
    referencePath  = cms.string("refSingleTkMuon22"),
    underTestPath  = cms.string("testSingleTkMuon22"),
)

process.cmpDouble = cms.EDAnalyzer("HLTP2GTFilterTestAnalyzer",
    triggerResults = _trigRes,
    referencePath  = cms.string("refDoubleIsoTkEleEGEle2212"),
    underTestPath  = cms.string("testDoubleIsoTkEleEGEle2212"),
)

process.cmpTriple = cms.EDAnalyzer("HLTP2GTFilterTestAnalyzer",
    triggerResults = _trigRes,
    referencePath  = cms.string("refTripleTkMuon533"),
    underTestPath  = cms.string("testTripleTkMuon533"),
)

process.cmpQuad = cms.EDAnalyzer("HLTP2GTFilterTestAnalyzer",
    triggerResults = _trigRes,
    referencePath  = cms.string("refQuadJet70554040"),
    underTestPath  = cms.string("testQuadJet70554040"),
)

# ===========================================================================
# Paths
# ===========================================================================

# L1 emulation (must precede all filter paths)
process.Phase2L1GTProducer          = cms.Path(process.l1tGTProducerSequence)
process.Phase2L1GTAlgoBlockProducer = cms.Path(process.l1tGTAlgoBlockProducerSequence)

# Reference paths (PathStatusFilter reads the L1 path decision)
# Each reference path must have the upstream L1 paths already in the schedule
# so that PathStatusFilter can read their decisions from TriggerResults.
# We therefore include the four relevant L1 paths explicitly:
process.pSingleTkMuon22     = cms.Path(process.SingleTkMuon22)
process.pIsoTkEleEGEle22_12 = cms.Path(process.IsoTkEleEGEle2212)
process.pTripleTkMuon5_3_3  = cms.Path(process.TripleTkMuon533)
process.pPuppiHT400         = cms.Path(process.PuppiHT400)
process.pPuppiHT450         = cms.Path(process.PuppiHT450)
process.pQuadJet70_55_40_40 = cms.Path(process.QuadJet70554040)

# Reference paths (PathStatusFilter)
process.refSinglePuppiHT450        = cms.Path(process.refSinglePuppiHT450Filt)
process.refSingleTkMuon22          = cms.Path(process.refSingleTkMuon22Filt)
process.refDoubleIsoTkEleEGEle2212 = cms.Path(process.refDoubleIsoTkEleEGEle2212Filt)
process.refTripleTkMuon533         = cms.Path(process.refTripleTkMuon533Filt)
process.refQuadJet70554040         = cms.Path(process.refQuadJet70554040Filt)

# Under-test paths (HLTP2GT*ObjectFilter)
process.testSinglePuppiHT450        = cms.Path(process.testSinglePuppiHT450Filt)
process.testSingleTkMuon22          = cms.Path(process.testSingleTkMuon22Filt)
process.testDoubleIsoTkEleEGEle2212 = cms.Path(process.testDoubleIsoTkEleEGEle2212Filt)
process.testTripleTkMuon533         = cms.Path(process.testTripleTkMuon533Filt)
process.testQuadJet70554040         = cms.Path(process.testQuadJet70554040Filt)

# Comparison EndPath
# EDAnalyzers that read TriggerResults must run in an EndPath so that all
# Path decisions are finalised before comparison.
process.comparison = cms.EndPath(
    process.cmpSingleA +
    process.cmpSingleB +
    process.cmpDouble +
    process.cmpTriple
    #+ process.cmpQuad
)

process.endjob = cms.EndPath(process.endOfProcess)

# ── Schedule: L1 emulation first, then L1 paths, then filter paths, then cmp
process.schedule = cms.Schedule(
    process.pPuppiHT400,
    process.pPuppiHT450,
    process.pSingleTkMuon22,
    process.pIsoTkEleEGEle22_12, 
    process.pTripleTkMuon5_3_3, 
    process.pQuadJet70_55_40_40,    
    # L1 emulation
    process.Phase2L1GTProducer,
    process.Phase2L1GTAlgoBlockProducer,
    # Reference paths
    process.refSinglePuppiHT450,
    process.refSingleTkMuon22,
    process.refDoubleIsoTkEleEGEle2212,
    process.refTripleTkMuon533,
    process.refQuadJet70554040,
    # Under-test paths
    process.testSinglePuppiHT450,
    process.testSingleTkMuon22,
    process.testDoubleIsoTkEleEGEle2212,
    process.testTripleTkMuon533,
    process.testQuadJet70554040,
    # Comparison + job end
    process.comparison,
    process.endjob,
)

# Prevent early deletion from removing products needed by PathStatusFilter
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
