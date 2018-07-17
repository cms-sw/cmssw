import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuBackgroundAnalysis")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/scratch1/users/fabozzi/patv2_skim/testSkim_v2.root"
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("test.root")
)

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20"),
    isoCut = cms.double(3.0),
    isolationType = cms.string("track")
)

process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

goodZToMuMuTemplate = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("replace this string with your cut"),
    src = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(False)
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


goodZToMuMuPlotsTemplate = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMu")
)

process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMu")
setattr(process, "nonIsolatedZToMuMuPlots", nonIsolatedZToMuMuPlots)

etaBounds = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]

def addModulesFromTemplate(sequence, moduleLabel, src, probeSelection):
    print "selection for: ", moduleLabel   
    for i in range(len(etaBounds)-1):
        etaMin = etaBounds[i]
        etaMax = etaBounds[i+1]
        module = copy.deepcopy(goodZToMuMuTemplate)
        if probeSelection == "single":
            cut = "%5.3f < daughter(1).eta < %5.3f" %(etaMin, etaMax)
        elif probeSelection == "double":
            cut = "%5.3f < daughter(0).eta < %5.3f | %5.3f < daughter(1).eta < %5.3f" %(etaMin, etaMax, etaMin, etaMax)
        print i, ") cut = ",  cut 
        setattr(module, "cut", cut)
        setattr(module, "src", cms.InputTag(src))
        copyModuleLabel = moduleLabel + str(i)
        setattr(process, copyModuleLabel, module)
        if sequence == None:
            sequence = module
        else:
            sequence = sequence + module
        plotModule = copy.deepcopy(goodZToMuMuPlotsTemplate)
        setattr(plotModule, "src", cms.InputTag(copyModuleLabel))
        for h in plotModule.histograms:
            h.description.setValue(h.description.value() + ": " + "#eta: [%5.3f, %5.3f]" %(etaMin, etaMax))
        plotModuleLabel = moduleLabel + "Plots" + str(i)
        setattr(process, plotModuleLabel, plotModule)
        sequence = sequence + plotModule
    path = cms.Path(sequence)
    setattr(process, moduleLabel+"Path", path)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
process.goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuon")
process.goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrack")

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMuPlots,
    "goodZToMuMu", "goodZToMuMu",
    "double")
    
process.nonIsolatedZToMuMuPath = cms.Path (
    process.nonIsolatedZToMuMu +
    process.nonIsolatedZToMuMuPlots,
)

addModulesFromTemplate(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonPlots, 
    "goodZToMuMuOneStandAloneMuon", "goodZToMuMuOneStandAloneMuon",
    "single")

addModulesFromTemplate(
    ~process.goodZToMuMu +
    ~process.zToMuMuOneStandAloneMuon +
    process.zToMuGlobalMuOneTrack +
    process.zToMuMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackPlots,
    "goodZToMuMuOneTrack", "goodZToMuMuOneTrack",
    "single")
  
process.endPath = cms.EndPath( 
    process.eventInfo 
)

