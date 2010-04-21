#analysis code.
#It produces plot for Fit study
#author Luca Lista
#modificated by Noli Pasquale 21-10-2008 

import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuAnalysis")

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
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_1.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_2.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_3.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_4.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_5.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_6.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_7.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_8.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_9.root",
    "rfio:/castor/cern.ch/user/n/noli/Summer08/reskimed/zmm/zmumu_reskim_10.root"
    
      )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("analysis_zmunu_reskim_10000evt.root")
)

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20"),
    isoCut = cms.double(3.0),
    isolationType = cms.string("track"),
 )

process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)
#ZMuMu: richiedo almeno 1 HLT trigger match.Per la shape
process.goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

#ZMuMu: richiedo 2 HLT trigger match
process.goodZToMuMu2HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

#ZMuMu: richiedo 1 HLT trigger match
process.goodZToMuMu1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)


process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

#ZMuMu1notIso: richiedo almeno un trigger
process.nonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("nonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
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

#ZMuTk:richiedo che il muGlobal 'First' ha HLT match
process.goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

#ZMuSta:richiedo che il muGlobal 'First' ha HLT match
process.goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

goodZToMuMuTemplate = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("replace this string with your cut"),
    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    filter = cms.bool(False)
)


#### Plot ###

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


#ZMuMu almeno 1 HLT + 2 track-iso (Shape)
goodZToMuMuPlotsTemplate = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLT")
)

#ZMuMu almeno 1 HLT + almeno 1 NON track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")
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

#ZMuMu almeno  1 HLT + 2  track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate

#ZMuMu 1 HLT + 2  track-iso
process.goodZToMuMu1HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMu1HLTPlots.src = cms.InputTag("goodZToMuMu1HLT")

#ZMuMu 2 HLT + 2  track-iso
process.goodZToMuMu2HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMu2HLTPlots.src = cms.InputTag("goodZToMuMu2HLT")

#ZMuSta First HLT + 2  track-iso
process.goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")

#ZMuTk First HLT + 2  track-iso
process.goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMuAtLeast1HLT+
    process.goodZToMuMuPlots,
    "goodZToMuMu", "goodZToMuMu",
    "double")

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMu2HLT+
    process.goodZToMuMu2HLTPlots,
    "goodZToMuMu2HLT", "goodZToMuMu2HLT",
    "double")

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMu1HLT+
    process.goodZToMuMu1HLTPlots,
    "goodZToMuMu1HLT", "goodZToMuMu1HLT",
    "double")
    
process.nonIsolatedZToMuMuPath = cms.Path (
    process.nonIsolatedZToMuMu +
    process.nonIsolatedZToMuMuAtLeast1HLT+
    process.nonIsolatedZToMuMuPlots
)

addModulesFromTemplate(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT+
    process.goodZToMuMuOneStandAloneMuonPlots, 
    "goodZToMuMuOneStandAloneMuon", "goodZToMuMuOneStandAloneMuon",
    "single")

addModulesFromTemplate(
    ~process.goodZToMuMu +
    ~process.zToMuMuOneStandAloneMuon +
    process.zToMuGlobalMuOneTrack +
    process.zToMuMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackFirstHLT+
    process.goodZToMuMuOneTrackPlots,
    "goodZToMuMuOneTrack", "goodZToMuMuOneTrack",
    "single")
  
process.endPath = cms.EndPath( 
    process.eventInfo 
)

