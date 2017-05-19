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
    input = cms.untracked.int32(1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
"rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_1.root"
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("zMuMu_qcd.root")
)

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.0 & abs(daughter(1).eta)<2.0  &  mass > 20"),
   # adjusting isolation values  
    isoCut = cms.double(3.0),
    ptThreshold = cms.untracked.double(1.5),
    etEcalThreshold = cms.untracked.double(0.2),
    etHcalThreshold = cms.untracked.double(0.5),
    deltaRTrk = cms.untracked.double(0.3),
    deltaREcal = cms.untracked.double(0.25),
    deltaRHcal = cms.untracked.double(0.25),
    # setting alpha=0 in order to have only tracker isolation
    alpha = cms.untracked.double(0),
    beta = cms.untracked.double(-0.75),
    relativeIsolation = cms.bool(False)
 )

#selecting endcap / endcap-barrel / barrel regions

dict = {'Barrel':[0, 0.8],'BarrEnd':[0.8, 1.2],'EndCap':[1.2, 2.0] }


def addModuleEtaRegions(moduleToModify, region, src="", cut =""):
    print "selection for: ", moduleToModify.label()+region   
    etaMin = dict[region][0]
    etaMax = dict[region][1]
    module = copy.deepcopy(moduleToModify)
    if cut=="":
        cut = "mass>20 & charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & ( abs(daughter(0).eta) > %5.3f & abs( daughter(0).eta )< %5.3f)  &  ( abs(daughter(1).eta) > %5.3f & abs( daughter(1).eta )< %5.3f) " %(etaMin, etaMax, etaMin, etaMax)
    print region, ") cut = ",  cut 
    if 'cut' in  module.parameters_():
        setattr(module, "cut", cut)
    if 'src' in   module.parameters_():
        setattr(module, "src", src)
    copyModuleLabel = moduleToModify.label() + region
    setattr(process, copyModuleLabel, module)
    return  module


def addPlotModuleEtaRegions(plotModuleToModify, region, src=""):       
    print "selection for: ", plotModuleToModify.label()
    etaMin = dict[region][0]
    etaMax = dict[region][1]
    plotModule = copy.deepcopy(plotModuleToModify)
    for h in plotModule.histograms:
            h.description.setValue(h.description.value() + ": " + "|#eta|: [%5.3f, %5.3f]" %(etaMin, etaMax))
    if 'src' in   plotModule.parameters_():
        setattr(plotModule, "src", src)      
    copyPlotModuleLabel = plotModuleToModify.label() + region
    setattr(process, copyPlotModuleLabel, plotModule)
    return  plotModule



process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

for region in dict.keys(): 
    addModuleEtaRegions(process.goodZToMuMu, region,"dimuonsGlobal"  )

#ZMuMu: richiedo almeno 1 HLT trigger match.Per la shape
process.goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

## for region in dict.keys():
##     addModuleEtaRegions(process.goodZToMuMuAtLeast1HLT, region, addModuleEtaRegions(process.goodZToMuMu, region,"dimuonsGlobal"  ) )



#ZMuMu: richiedo 2 HLT trigger match
process.goodZToMuMu2HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

## for region in dict.keys(): 
##     addModuleEtaRegions(process.goodZToMuMu2HLT, region,addModuleEtaRegions(process.goodZToMuMu, region,"dimuonsGlobal"  ) )




#ZMuMu: richiedo 1 HLT trigger match
process.goodZToMuMu1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)
## for region in dict.keys(): 
##     addModuleEtaRegions(process.goodZToMuMu1HLT, region,addModuleEtaRegions(process.goodZToMuMu, region,"dimuonsGlobal"  ) )



process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

for region in dict.keys(): 
    addModuleEtaRegions(process.nonIsolatedZToMuMu, region,"dimuonsGlobal" )



#ZMuMu1notIso: richiedo almeno un trigger
process.nonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("nonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)
## for region in dict.keys(): 
##     addModuleEtaRegions(process.nonIsolatedZToMuMuAtLeast1HLT, region,  addModuleEtaRegions(process.nonIsolatedZToMuMu, region,"dimuonsGlobal" ))


process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)
 ## for region in dict.keys(): 
##      addModuleEtaRegions(process.zToMuGlobalMuOneTrack, region, "dimuonsOneTrack" ,"daughter(0).isGlobalMuon = 1")


process.zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)
for region in dict.keys(): 
    addModuleEtaRegions(process.zToMuMuOneTrack, region, "zToMuGlobalMuOneTrack" )


process.zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)

for region in dict.keys(): 
    addModuleEtaRegions(process.zToMuMuOneStandAloneMuon, region, "dimuonsOneStandAloneMuon" )


process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrackBarrel = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrackBarrel"),
    overlap = cms.InputTag("goodZToMuMuBarrel"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrackEndCap = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrackEndCap"),
    overlap = cms.InputTag("goodZToMuMuEndCap"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrackBarrEnd = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrackBarrEnd"),
    overlap = cms.InputTag("goodZToMuMuBarrEnd"),
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

process.goodZToMuMuOneTrackFirstHLTBarrel = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackBarrel"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)

process.goodZToMuMuOneTrackFirstHLTEndCap = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackEndCap"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)
process.goodZToMuMuOneTrackFirstHLTBarrEnd = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackBarrEnd"),
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

## for region in dict.keys():
##      addModuleEtaRegions(process.goodZToMuMuOneStandAloneMuon, region,  addModuleEtaRegions(process.zToMuMuOneStandAloneMuon, region, "dimuonsOneStandAloneMuon"  ))

#ZMuSta:richiedo che il muGlobal 'First' ha HLT match
process.goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)
## for region in dict.keys():
##    addModuleEtaRegions(process.goodZToMuMuOneStandAloneMuonFirstHLT, region,addModuleEtaRegions(process.goodZToMuMuOneStandAloneMuon, region,  addModuleEtaRegions(process.zToMuMuOneStandAloneMuon, region, "dimuonsOneStandAloneMuon"  )) )


process.zmumuSaMassHistogram = cms.EDAnalyzer(
    "ZMuMuSaMassHistogram",
    src_m = cms.InputTag("goodZToMuMu"),
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbin = cms.untracked.int32(200)
#   name = cms.untracked.string("zMass")    
    )

for region in dict.keys():
    addModuleEtaRegions(process.zmumuSaMassHistogram, region, addModuleEtaRegions(process.goodZToMuMu, region,"dimuonsGlobal"  )  )   



goodZToMuMuTemplate = cms.EDFilter(
   "CandViewRefSelector",
    cut = cms.string(""),
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(False)
)

### Plot ###

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
for region in dict.keys():
     addPlotModuleEtaRegions(process.goodZToMuMuPlots, region, "goodZToMuMuAtLeast1HLT" )   


nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")
setattr(process, "nonIsolatedZToMuMuPlots", nonIsolatedZToMuMuPlots)
for region in dict.keys():
     addPlotModuleEtaRegions(process.nonIsolatedZToMuMuPlots, region, "nonIsolatedZToMuMuAtLeast1HLT" )   





process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

#ZMuMu almeno  1 HLT + 2  track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate

#ZMuMu 1 HLT + 2  track-iso
process.goodZToMuMu1HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMu1HLTPlots.src = cms.InputTag("goodZToMuMu1HLT")

for region in dict.keys():
     addPlotModuleEtaRegions(process.goodZToMuMu1HLTPlots, region, "goodZToMuMuAtLeast1HLT" )   

#ZMuMu 2 HLT + 2  track-iso
process.goodZToMuMu2HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMu2HLTPlots.src = cms.InputTag("goodZToMuMu2HLT")
for region in dict.keys():
     addPlotModuleEtaRegions(process.goodZToMuMu2HLTPlots, region, "goodZToMuMuAtLeast1HLT" )   


#ZMuSta First HLT + 2  track-iso
process.goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")
for region in dict.keys():
     addPlotModuleEtaRegions(process.goodZToMuMuOneStandAloneMuonPlots, region, "goodZToMuMuOneStandAloneMuonFirstHLT" )   




#ZMuTk First HLT + 2  track-iso
process.goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")

process.goodZToMuMuOneTrackPlotsBarrel = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlotsBarrel.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTBarrel")

process.goodZToMuMuOneTrackPlotsEndCap = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlotsEndCap.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTEndCap")
 
process.goodZToMuMuOneTrackPlotsBarrEnd = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuOneTrackPlotsBarrEnd.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTBarrEnd")


# N-tuples

process.goodZToMuMuOneStandAloneMuonNtuple = cms.EDProducer(
    "CandViewNtpProducer",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    variables = cms.VPSet(
      cms.PSet(
        tag = cms.untracked.string("mass"),
        quantity = cms.untracked.string("mass")
      )
    )
)

process.initialGoodZToMuMuPath = cms.Path( 
    process.goodZToMuMu +
    process.zmumuSaMassHistogram     
)

process.initialGoodZToMuMuBarrelPath = cms.Path( 
    process.goodZToMuMuBarrel+
    process.zmumuSaMassHistogramBarrel     
)

process.initialGoodZToMuMuEndCapPath = cms.Path( 
    process.goodZToMuMuEndCap+
    process.zmumuSaMassHistogramEndCap     
)

process.initialGoodZToMuMuBarEndCapPath = cms.Path( 
    process.goodZToMuMuBarrEnd+
    process.zmumuSaMassHistogramBarrEnd     
)



process.goodZToMuMuFinalPath = cms.Path (
     process.goodZToMuMu +
     process.goodZToMuMuAtLeast1HLT+
     process.goodZToMuMuPlots
     )

process.goodZToMuMuFinalBarrelPath = cms.Path (
     process.goodZToMuMuBarrel +
     process.goodZToMuMuAtLeast1HLT+
     process.goodZToMuMuPlotsBarrel
     )

process.goodZToMuMuFinalEndCapPath = cms.Path (
     process.goodZToMuMuEndCap +
     process.goodZToMuMuAtLeast1HLT+
     process.goodZToMuMuPlotsEndCap
     )

process.goodZToMuMuFinalBarrEndPath = cms.Path (
     process.goodZToMuMuBarrEnd +
     process.goodZToMuMuAtLeast1HLT+
     process.goodZToMuMuPlotsBarrEnd
     )

process.goodZToMuMu2HLTFinalPath= cms.Path( 
     process.goodZToMuMu +
     process.goodZToMuMu2HLT +
     process.goodZToMuMu2HLTPlots
     )


process.goodZToMuMu2HLTBarrelFinalPath= cms.Path( 
     process.goodZToMuMuBarrel +
     process.goodZToMuMu2HLT +
     process.goodZToMuMu2HLTPlotsBarrel
     )

process.goodZToMuMu2HLTEndCapFinalPath= cms.Path( 
     process.goodZToMuMuEndCap+
     process.goodZToMuMu2HLT +
     process.goodZToMuMu2HLTPlotsEndCap
     )

process.goodZToMuMu2HLTBarrEndFinalPath= cms.Path( 
     process.goodZToMuMuBarrEnd +
     process.goodZToMuMu2HLT +
     process.goodZToMuMu2HLTPlotsBarrEnd
     )


process.goodZToMuMu1HLTFinalPath= cms.Path( 
     process.goodZToMuMu +
     process.goodZToMuMu1HLT +
     process.goodZToMuMu1HLTPlots
     )


process.goodZToMuMu1HLTBarrelFinalPath= cms.Path( 
     process.goodZToMuMuBarrel +
     process.goodZToMuMu1HLT +
     process.goodZToMuMu1HLTPlotsBarrel
     )

process.goodZToMuMu1HLTEndCapFinalPath= cms.Path( 
     process.goodZToMuMuEndCap+
     process.goodZToMuMu1HLT +
     process.goodZToMuMu1HLTPlotsEndCap
     )

process.goodZToMuMu1HLTBarrEndFinalPath= cms.Path( 
     process.goodZToMuMuBarrEnd +
     process.goodZToMuMu1HLT +
     process.goodZToMuMu1HLTPlotsBarrEnd
     )



process.nonIsolatedZToMuMuFinalPath = cms.Path (
     process.nonIsolatedZToMuMu +
     process.nonIsolatedZToMuMuAtLeast1HLT +
     process.nonIsolatedZToMuMuPlots
 )

process.nonIsolatedZToMuMuFinalBarrelPath = cms.Path (
     process.nonIsolatedZToMuMuBarrel +
     process.nonIsolatedZToMuMuAtLeast1HLT +
     process.nonIsolatedZToMuMuPlotsBarrel
 )

process.nonIsolatedZToMuMuFinalEndCapPath = cms.Path (
     process.nonIsolatedZToMuMuEndCap +
     process.nonIsolatedZToMuMuAtLeast1HLT +
     process.nonIsolatedZToMuMuPlotsEndCap
 )

process.nonIsolatedZToMuMuFinalBarrEndPath = cms.Path (
     process.nonIsolatedZToMuMuBarrEnd +
     process.nonIsolatedZToMuMuAtLeast1HLT +
     process.nonIsolatedZToMuMuPlotsBarrEnd
 )




process.goodZToMuMuOneStandAloneMuonFinalPath=cms.Path(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.goodZToMuMuOneStandAloneMuonNtuple +
    process.goodZToMuMuOneStandAloneMuonPlots
   )

process.goodZToMuMuOneStandAloneMuonFinalBarrelPath=cms.Path(
    ~process.goodZToMuMuBarrel + 
    process.zToMuMuOneStandAloneMuonBarrel + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.goodZToMuMuOneStandAloneMuonNtuple +
    process.goodZToMuMuOneStandAloneMuonPlotsBarrel
   )   

process.goodZToMuMuOneStandAloneMuonFinalEndCapPath=cms.Path(
    ~process.goodZToMuMuEndCap + 
    process.zToMuMuOneStandAloneMuonEndCap + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.goodZToMuMuOneStandAloneMuonNtuple +
    process.goodZToMuMuOneStandAloneMuonPlotsEndCap
   )

process.goodZToMuMuOneStandAloneMuonFinalBarrEndPath=cms.Path(
    ~process.goodZToMuMuBarrEnd + 
    process.zToMuMuOneStandAloneMuonBarrEnd + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.goodZToMuMuOneStandAloneMuonNtuple +
    process.goodZToMuMuOneStandAloneMuonPlotsBarrEnd
   )   




process.goodZToMuMuOneTrackFinalPath=cms.Path(
     ~process.goodZToMuMu +
     ~process.zToMuMuOneStandAloneMuon +
     process.zToMuGlobalMuOneTrack +
     process.zToMuMuOneTrack +
     process.goodZToMuMuOneTrack +
     process.goodZToMuMuOneTrackFirstHLT +
     process.goodZToMuMuOneTrackPlots
     )

process.goodZToMuMuOneTrackFinalBarrelPath=cms.Path(
     ~process.goodZToMuMuBarrel +
     ~process.zToMuMuOneStandAloneMuonBarrel +
     process.zToMuGlobalMuOneTrack +
     process.zToMuMuOneTrackBarrel +
     process.goodZToMuMuOneTrackBarrel +
     process.goodZToMuMuOneTrackFirstHLTBarrel +
     process.goodZToMuMuOneTrackPlotsBarrel
     )

process.goodZToMuMuOneTrackFinalEndCapPath=cms.Path(
     ~process.goodZToMuMuEndCap +
     ~process.zToMuMuOneStandAloneMuonEndCap +
     process.zToMuGlobalMuOneTrack +
     process.zToMuMuOneTrackEndCap +
     process.goodZToMuMuOneTrackEndCap +
     process.goodZToMuMuOneTrackFirstHLTEndCap +
     process.goodZToMuMuOneTrackPlotsEndCap
     )

process.goodZToMuMuOneTrackFinalBarrEndPath=cms.Path(
     ~process.goodZToMuMuBarrEnd +
     ~process.zToMuMuOneStandAloneMuonBarrEnd +
     process.zToMuGlobalMuOneTrack +
     process.zToMuMuOneTrackBarrEnd +
     process.goodZToMuMuOneTrackBarrEnd +
     process.goodZToMuMuOneTrackFirstHLTBarrEnd +
     process.goodZToMuMuOneTrackPlotsBarrEnd
     )







process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('file:./zMuSa-UML.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*"
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuOneStandAloneMuonFinalPath"
      )
    )
)
  
process.endPath = cms.EndPath( 
    process.eventInfo  +
    process.out
)

