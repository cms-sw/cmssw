import FWCore.ParameterSet.Config as cms

# reorganization of Z->mumu categories sequence, to run after the ZMuMu(Sub)Skim (i.d. supposing dimuons, dimuonsGlobal, dimuonsOneTrack and dimuonsOneStndAloneMuon categories has been built)


### parameter set to be overloaded in the configuration file 


#from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff import *

from ElectroWeakAnalysis.ZMuMu.goodZToMuMu_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuSameCharge_cfi import *
from ElectroWeakAnalysis.ZMuMu.nonIsolatedZToMuMu_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuOneTrack_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuOneTrackerMuon_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuOneStandAloneMuon_cfi import *

### for zmusta modelling...

zmumuSaMassHistogram = cms.EDAnalyzer(
    "ZMuMuSaMassHistogram",
    src_m = cms.InputTag("goodZToMuMu"),
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbin = cms.untracked.int32(200)
   # name = cms.untracked.string("zMass")    
    )

### Primary vertex info

eventVtxInfoNtuple = cms.EDProducer(
    "EventVtxInfoNtupleDumper",
    primaryVertices=cms.InputTag("offlinePrimaryVertices")
)

# path for dumping vtx info in the ntuple
generalEventInfoPath = cms.Path(
    eventVtxInfoNtuple
    )


### paths for loose cuts, not notIso ones, not 1HLT and 2HLT: only ZGolden, zMuSta, zMuTk, zMuTrackerMuon and ZGoldenSameCharge..

goodZToMuMuPathLoose = cms.Path(
    
    goodZToMuMuLoose +
    goodZToMuMuAtLeast1HLTLoose
    )



goodZToMuMu2HLTPathLoose = cms.Path(
    goodZToMuMuLoose +
    goodZToMuMu2HLTLoose
    )

goodZToMuMu1HLTPathLoose = cms.Path(
    goodZToMuMuLoose +
    goodZToMuMu1HLTLoose
    )

goodZToMuMuAB1HLTPathLoose=cms.Path(
    goodZToMuMuNotFiltered+  ## not filtered
    zToMuMuABLoose+
    goodZToMuMuABLoose+
    goodZToMuMuAB1HLTLoose
)

goodZToMuMuBB2HLTPathLoose=cms.Path(
    zToMuMuBBLoose+
    goodZToMuMuBB2HLTLoose 
)

goodZToMuMuSameChargePathLoose = cms.Path(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameChargeLoose +
    goodZToMuMuSameChargeAtLeast1HLTLoose
    )


## goodZToMuMuSameCharge2HLTPathLoose = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameChargeLoose +
##     goodZToMuMuSameCharge2HLTLoose
##     )


## goodZToMuMuSameCharge1HLTPathLoose = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameChargeLoose +
##     goodZToMuMuSameCharge1HLTLoose
##     )



goodZToMuMuOneStandAloneMuonPathLoose = cms.Path(
### I should deny the tight zmumu, otherwise I cut to much.... 
    ~goodZToMuMu + 
    zToMuMuOneStandAloneMuonLoose + 
    goodZToMuMuOneStandAloneMuonLoose +
    goodZToMuMuOneStandAloneMuonFirstHLTLoose 
    )


goodZToMuMuOneTrackerMuonPathLoose= cms.Path(
    ### I should deny the tight zmumu, otherwise I cut to much.... 
    ~goodZToMuMu +
    zToMuMuOneTrackerMuonLoose + 
    goodZToMuMuOneTrackerMuonLoose +
    goodZToMuMuOneTrackerMuonFirstHLTLoose 
)



goodZToMuMuOneTrackPathLoose=cms.Path(
    ### I should deny the tight zmumu, otherwise I cut to much.... 
    ~goodZToMuMu +
    ~zToMuMuOneStandAloneMuon +
    zToMuGlobalMuOneTrack +
    zToMuMuOneTrackLoose +
    goodZToMuMuOneTrackLoose +
    goodZToMuMuOneTrackFirstHLTLoose 
    )





### sequences and path for tight cuts...


globalMuQualityCutsAnalysisAA= cms.EDAnalyzer(
    "GlbMuQualityCutsAnalysis",
    src = cms.InputTag("goodZToMuMu"), 
    ptMin = cms.untracked.double(0.0),
    massMin = cms.untracked.double(0.0),
    massMax = cms.untracked.double(200.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(10.0),
    trkIso = cms.untracked.double(10000),
    chi2Cut = cms.untracked.double(10),
    nHitCut = cms.untracked.int32(10)
 )

globalMuQualityCutsAnalysisAB= cms.EDAnalyzer(
    "GlbMuQualityCutsAnalysis",
    src = cms.InputTag("goodZToMuMuAB"), 
    ptMin = cms.untracked.double(0.0),
    massMin = cms.untracked.double(0.0),
    massMax = cms.untracked.double(200.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(10.0),
    trkIso = cms.untracked.double(10000),
    chi2Cut = cms.untracked.double(10),
    nHitCut = cms.untracked.int32(10)
 )

globalMuQualityCutsAnalysisAAtrk= cms.EDAnalyzer(
    "GlbMuQualityCutsAnalysis",
    src = cms.InputTag("goodZToMuMuOneTrackerMuon"), 
    ptMin = cms.untracked.double(0.0),
    massMin = cms.untracked.double(0.0),
    massMax = cms.untracked.double(200.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(10.0),
    trkIso = cms.untracked.double(10000),
    chi2Cut = cms.untracked.double(10),
    nHitCut = cms.untracked.int32(10)
 )

globalMuQualityCutsAnalysisAAsta= cms.EDAnalyzer(
    "GlbMuQualityCutsAnalysis",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"), 
    ptMin = cms.untracked.double(0.0),
    massMin = cms.untracked.double(0.0),
    massMax = cms.untracked.double(200.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(10.0),
    trkIso = cms.untracked.double(10000),
    chi2Cut = cms.untracked.double(10),
    nHitCut = cms.untracked.int32(10)
 )


initialGoodZToMuMuPath = cms.Path( 
    goodZToMuMu +
    zmumuSaMassHistogram     
)


goodZToMuMuPath = cms.Path(
    goodZToMuMu +
    goodZToMuMuAtLeast1HLT
  ##  globalMuQualityCutsAnalysisAA 
    )



goodZToMuMu2HLTPath = cms.Path(
    goodZToMuMu +
    goodZToMuMu2HLT
    )


goodZToMuMu1HLTPath = cms.Path(
    goodZToMuMu +
    goodZToMuMu1HLT
    )

goodZToMuMuAB1HLTPath=cms.Path(
    goodZToMuMuNotFiltered + ## not filtered
    zToMuMuAB+
    goodZToMuMuAB+
    goodZToMuMuAB1HLT
##    globalMuQualityCutsAnalysisAB
)

goodZToMuMuBB2HLTPath=cms.Path(
    zToMuMuBB+
    goodZToMuMuBB2HLT 
)


goodZToMuMuSameChargePath = cms.Path(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameChargeAtLeast1HLT
    )


goodZToMuMuSameCharge2HLTPath = cms.Path(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameCharge2HLT
    )



goodZToMuMuSameCharge1HLTPath = cms.Path(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameCharge1HLT
    )



nonIsolatedZToMuMuPath = cms.Path (
    nonIsolatedZToMuMu +
    nonIsolatedZToMuMuAtLeast1HLT 
)


oneNonIsolatedZToMuMuPath  = cms.Path(
    nonIsolatedZToMuMu  +
    oneNonIsolatedZToMuMu  +
    oneNonIsolatedZToMuMuAtLeast1HLT  
)


twoNonIsolatedZToMuMuPath  = cms.Path(
    nonIsolatedZToMuMu  +
    twoNonIsolatedZToMuMu  +
    twoNonIsolatedZToMuMuAtLeast1HLT  
)


goodZToMuMuOneStandAloneMuonPath = cms.Path(
    ~goodZToMuMu +
    zToMuMuOneStandAloneMuon + 
    goodZToMuMuOneStandAloneMuon +
    goodZToMuMuOneStandAloneMuonFirstHLT 
##    globalMuQualityCutsAnalysisAAsta
    )

goodZToMuMuOneTrackerMuonPath= cms.Path(
    ~goodZToMuMu +
    zToMuMuOneTrackerMuon + 
    goodZToMuMuOneTrackerMuon +
    goodZToMuMuOneTrackerMuonFirstHLT 
##    globalMuQualityCutsAnalysisAAtrk
)



goodZToMuMuOneTrackPath=cms.Path(
    ~goodZToMuMu +
    ~zToMuMuOneStandAloneMuon +
    zToMuGlobalMuOneTrack +
    zToMuMuOneTrack +
    goodZToMuMuOneTrack +
    goodZToMuMuOneTrackFirstHLT 
    )

###### endPath




eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

eventInfo.setLabel("eventInfo")

NtuplesOut = cms.Sequence(
    eventInfo
    )


VtxedNtuplesOut = cms.Sequence(
    eventInfo
    )



endPath = cms.EndPath( 
   NtuplesOut +
   VtxedNtuplesOut 
)


