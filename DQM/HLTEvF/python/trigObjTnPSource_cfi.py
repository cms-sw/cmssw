import FWCore.ParameterSet.Config as cms

_ecalBarrelEtaCut = cms.PSet(
    rangeVar = cms.string("eta"),
    allowedRanges=cms.vstring("-1.4442:1.4442")
    )
_ecalEndcapEtaCut = cms.PSet(
    rangeVar = cms.string("eta"),
    allowedRanges=cms.vstring("-2.5:-1.556","1.556:2.5")
    )

_ecalEtaCut = cms.PSet(
    rangeVar = cms.string("eta"),
    allowedRanges=cms.vstring("-2.5:-1.556","-1.4442:1.4442","1.556:2.5")
    )

_ptEBHist = cms.PSet(
    nameSuffex = cms.string("_ptEB"),
    titleSuffex = cms.string(" (Barrel);p_{T} [GeV];mass [GeV]"),
    bins = cms.vdouble(32,40,50,100),
    filler = cms.PSet(var = cms.string("pt"),localCuts = cms.VPSet(_ecalBarrelEtaCut))
    )
_ptEEHist = cms.PSet(
    nameSuffex = cms.string("_ptEE"),
    titleSuffex = cms.string(" (Endcap);p_{T} [GeV];mass [GeV]"),
    bins = cms.vdouble(32,40,50,100),
    filler = cms.PSet(var = cms.string("pt"),localCuts = cms.VPSet(_ecalEndcapEtaCut))
    )
_phiEBHist = cms.PSet(
    nameSuffex = cms.string("_phiEB"),
    titleSuffex = cms.string(" (Barrel);#phi [rad];mass [GeV]"),
    bins = cms.vdouble(-3.14,-1.57,0,1.57,3.14),
    filler = cms.PSet(var = cms.string("phi"),localCuts = cms.VPSet(_ecalBarrelEtaCut))
    )
_phiEEHist = cms.PSet(
    nameSuffex = cms.string("_phiEE"),
    titleSuffex = cms.string(" (Endcap);#phi [rad];mass [GeV]"),
    bins = cms.vdouble(-3.14,-1.57,0,1.57,3.14),
    filler = cms.PSet(var = cms.string("phi"),localCuts = cms.VPSet(_ecalEndcapEtaCut))
    )
_etaHist = cms.PSet(
    nameSuffex = cms.string("_eta"),
    titleSuffex = cms.string(";#eta;mass [GeV]"),
    bins = cms.vdouble(-2.5,-1.5,0,1.5,2.5),
    filler = cms.PSet(var = cms.string("eta"),localCuts = cms.VPSet())
    )

trigObjTnPSource = cms.EDAnalyzer('TrigObjTnPSource',
  triggerEvent = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  triggerResults = cms.InputTag('TriggerResults','','HLT'),
  histColls = cms.VPSet(cms.PSet(
    tagCuts = cms.VPSet(_ecalBarrelEtaCut),
    probeCuts = cms.VPSet(_ecalEtaCut),
    tagFilters = cms.PSet(
      filterSets = cms.VPSet(
        cms.PSet( 
           filters = cms.vstring(
             "hltEle32WPTightGsfTrackIsoFilter"
           ),
           isAND = cms.bool(False)
        ),
      ),
      isAND = cms.bool(False)
    ),
    collName = cms.string("stdTag"),
    folderName = cms.string("HLT/EGM/TrigObjTnP"),
    evtTrigSel = cms.PSet(
      selectionStr = cms.string("HLT_Ele32_WPTight_Gsf_v*"),
      isANDForExpandedPaths = cms.bool(False),
      verbose = cms.int32(1)
    ),
    histDefs = cms.PSet(
      massBins = cms.vdouble(i for i in range(60,120+1)),
      configs = cms.VPSet(_ptEBHist,_ptEEHist,_phiEBHist,_phiEEHist,_etaHist)
    ),
    probeFilters = cms.vstring("hltEG32L1SingleEGOrEtFilter",
                               "hltEle32WPTightClusterShapeFilter",
                               "hltEle32WPTightHEFilter",
                               "hltEle32WPTightEcalIsoFilter",
                               "hltEle32WPTightHcalIsoFilter",
                               "hltEle32WPTightPixelMatchFilter",
                               "hltEle32WPTightPMS2Filter",
                               "hltEle32WPTightGsfOneOEMinusOneOPFilter",
                               "hltEle32WPTightGsfMissingHitsFilter",
                               "hltEle32WPTightGsfDetaFilter",
                               "hltEle32WPTightGsfDphiFilter",
                               "hltEle32WPTightGsfTrackIsoFilter")           
  ))
)
