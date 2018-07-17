#analysis code.
#It produces plot for Fit study
#author Luca Lista
#modificated by Noli Pasquale 21-10-2008 
#modified by Annapaola de Cosa 18-12-2008
#modified by Michele de Gruttola 08-10-2009


import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuAnalysis")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(17810)
    input = cms.untracked.int32(-1)
)


process.load("Configuration.StandardSequences.MagneticField_cff")



#process.load("ElectroWeakAnalysis/ZMuMu/OCTSUBSKIM_cff")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_1.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_2.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_3.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_4.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_5.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_6.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_7.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_8.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_9.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_10.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_11.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_12.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_13.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_14.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_15.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_16.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_17.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_18.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_19.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_20.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_21.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_22.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_23.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_24.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_25.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_26.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_27.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_28.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_29.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_30.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_31.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_32.root", 
# "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_33.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_34.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_35.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_36.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_37.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_38.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_39.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_40.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_41.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_42.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_43.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_44.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_45.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_46.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_47.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_48.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_49.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_50.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_51.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_52.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_53.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_54.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_55.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_56.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_57.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_58.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_59.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_60.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_61.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_62.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_63.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_64.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_65.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_66.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_67.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_68.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_69.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_70.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_71.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_72.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_73.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_74.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_75.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_76.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_77.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_78.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_79.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_80.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_81.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_82.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_83.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_84.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_85.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_86.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_87.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_88.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_89.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_90.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_91.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_92.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_93.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_94.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_95.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_96.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_97.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_98.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_99.root"
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("Analysis_zmm_10TeV_SameCharge.root")
)

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 10 & daughter(1).pt > 10 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 0"),
    isoCut = cms.double(3.),
    ptThreshold = cms.untracked.double(1.5),
    etEcalThreshold = cms.untracked.double(0.2),
    etHcalThreshold = cms.untracked.double(0.5),
    deltaRVetoTrk = cms.untracked.double(0.015),
    deltaRTrk = cms.untracked.double(0.3),
    deltaREcal = cms.untracked.double(0.25),
    deltaRHcal = cms.untracked.double(0.25),
    alpha = cms.untracked.double(0.),
    beta = cms.untracked.double(-0.75),
    relativeIsolation = cms.bool(False)

# For standard isolation (I_Tkr<3GeV) choose this configuration:
#   isoCut = cms.double(3.),
#   ptThreshold = cms.untracked.double(1.5),
#   etEcalThreshold = cms.untracked.double(0.2),
#   etHcalThreshold = cms.untracked.double(0.5),
#   deltaRVetoTrk = cms.untracked.double(0.015),
#   deltaRTrk = cms.untracked.double(0.3),
#   deltaREcal = cms.untracked.double(0.25),
#   deltaRHcal = cms.untracked.double(0.25),
#   alpha = cms.untracked.double(0.),
#   beta = cms.untracked.double(-0.75),
#   relativeIsolation = cms.bool(False)
 )


process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)



# same charge dimuons....
process.dimuonsGlobalSameCharge = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('mass > 20 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)

process.goodZToMuMuSameCharge = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobalSameCharge"),
    filter = cms.bool(True) 
)
process.goodZToMuMuSameCharge.cut=cms.string("charge!=0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 20")


#ZMuMu: requiring at least  1 HLT trigger match (for the shape)
process.goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)




#ZMuMu: requiring  2 HLT trigger match
process.goodZToMuMu2HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMu: requiring 1 HLT trigger match
process.goodZToMuMu1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMu:at least one muon is not isolated 
process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

#ZMuMu:1 muon is not isolated 
process.oneNonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuOneNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("nonIsolatedZToMuMu"),
    filter = cms.bool(True) 
)

#ZMuMu: 2 muons are not isolated 
process.twoNonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuTwoNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("nonIsolatedZToMuMu"),
    filter = cms.bool(True) 
)



#ZMuMunotIso: requiring at least 1 trigger
process.nonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("nonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMuOnenotIso: requiring at least 1 trigger
process.oneNonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("oneNonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMuTwonotIso: requiring at least 1 trigger
process.twoNonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("twoNonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)



process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
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

#ZMuTk:requiring that the GlobalMuon 'First' has HLT match
process.goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

#ZMuSta:requiring that the GlobalMuon has HLT match
process.goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

process.zmumuSaMassHistogram = cms.EDAnalyzer(
    "ZMuMuSaMassHistogram",
    src_m = cms.InputTag("goodZToMuMu"),
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbin = cms.untracked.int32(200)
   # name = cms.untracked.string("zMass")    
    )

goodZToMuMuTemplate = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("replace this string with your cut"),
    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    filter = cms.bool(False)
)



#### Plot ####
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
    max = cms.untracked.double(2000.0),
    nbins = cms.untracked.int32(2000),
    name = cms.untracked.string("zMassUpToTwoTeV"),
    description = cms.untracked.string("Z mass [GeV/c^{2}]"),
    plotquantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    min = cms.untracked.double(-10.0),
    max = cms.untracked.double(10.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zEta"),
    description = cms.untracked.string("Z #eta"),
    plotquantity = cms.untracked.string("eta")
    ),
    cms.PSet(
    min = cms.untracked.double(-6.0),
    max = cms.untracked.double(6.0),
    nbins = cms.untracked.int32(120),
    name = cms.untracked.string("zRapidity"),
    description = cms.untracked.string("Z y"),
    plotquantity = cms.untracked.string("rapidity")
    ),
    cms.PSet(
    min = cms.untracked.double(0),
    max = cms.untracked.double(200),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zPt"),
    description = cms.untracked.string("Z p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("pt")
    ),
    cms.PSet(
    min = cms.untracked.double(-4),
    max = cms.untracked.double(4),
    nbins = cms.untracked.int32(80),
    name = cms.untracked.string("zPhi"),
    description = cms.untracked.string("Z #phi"),
    plotquantity = cms.untracked.string("phi")
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
    plotquantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)"),
    ),
    cms.PSet(
    min = cms.untracked.double(-6.0),
    max = cms.untracked.double(6.0),
    nbins = cms.untracked.int32(120),
    name = cms.untracked.string("mu1Eta"),
    description = cms.untracked.string("muon1 #eta"),
    plotquantity = cms.untracked.string("daughter(0).eta"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-6.0),
    max = cms.untracked.double(6.0),
    nbins = cms.untracked.int32(120),
    name = cms.untracked.string("mu2Eta"),
    description = cms.untracked.string("muon2 #eta"),
    plotquantity = cms.untracked.string("daughter(1).eta"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-6.0),
    max = cms.untracked.double(6.0),
    nbins = cms.untracked.int32(120),
    name = cms.untracked.string("mu1y"),
    description = cms.untracked.string("muon1 y"),
    plotquantity = cms.untracked.string("daughter(0).rapidity"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-6.0),
    max = cms.untracked.double(6.0),
    nbins = cms.untracked.int32(120),
    name = cms.untracked.string("mu2y"),
    description = cms.untracked.string("muon2 y"),
    plotquantity = cms.untracked.string("daughter(1).rapidity"),   
    ),
cms.PSet(
    min = cms.untracked.double(-4.0),
    max = cms.untracked.double(4.0),
    nbins = cms.untracked.int32(80),
    name = cms.untracked.string("mu1phi"),
    description = cms.untracked.string("muon1 #phi"),
    plotquantity = cms.untracked.string("daughter(0).phi"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-4.0),
    max = cms.untracked.double(4.0),
    nbins = cms.untracked.int32(80),
    name = cms.untracked.string("mu2phi"),
    description = cms.untracked.string("muon2 #phi"),
    plotquantity = cms.untracked.string("daughter(1).phi"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-0.1),
    max = cms.untracked.double(6.9),
    nbins = cms.untracked.int32(7000),
    name = cms.untracked.string("absMu1phiMinusMu2phi"),
    description = cms.untracked.string("|mu1 #phi - mu2 #phi|"),
    plotquantity = cms.untracked.string("abs(daughter(0).phi - daughter(1).phi)"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-10),
    max = cms.untracked.double(10),
    nbins = cms.untracked.int32(1000),
    name = cms.untracked.string("mu1 dxy"),
    description = cms.untracked.string("muon1 dxy"),
    plotquantity = cms.untracked.string("(- daughter(0).vx * daughter(0).py + daughter(0).vy * daughter(0).px) / daughter(0).pt "),   
    ),
    cms.PSet(
    min = cms.untracked.double(-10),
    max = cms.untracked.double(10),
    nbins = cms.untracked.int32(1000),
    name = cms.untracked.string("mu2 dxy"),
    description = cms.untracked.string("muon2 dxy"),
    plotquantity = cms.untracked.string("(- daughter(1).vx * daughter(1).py + daughter(1).vy * daughter(1).px) / daughter(1).pt "),   
    ),
    cms.PSet(
    min = cms.untracked.double(-10),
    max = cms.untracked.double(10),
    nbins = cms.untracked.int32(1000),
    name = cms.untracked.string("mu1 dz"),
    description = cms.untracked.string("muon1 dz"),
    plotquantity = cms.untracked.string("daughter(0).vz -  ( daughter(0).vx * daughter(0).px  + daughter(0).vy * daughter(0).py) / daughter(0).pt *  daughter(0).pz / daughter(0).pt"),   
    ),
    cms.PSet(
    min = cms.untracked.double(-10),
    max = cms.untracked.double(10),
    nbins = cms.untracked.int32(1000),
    name = cms.untracked.string("mu2 dz"),
    description = cms.untracked.string("muon2 dz"),
    plotquantity = cms.untracked.string("daughter(1).vz -  ( daughter(1).vx * daughter(1).px  + daughter(1).vy * daughter(1).py) / daughter(1).pt *  daughter(1).pz / daughter(1).pt"),   
    ) 
    )
)

## # dxy constructed from the vtx position
## dxy0 = " (- daughter(0).vx * daughter(0).py + daughter(0).vy * daughter(0).px) / daughter(0).pt "
## dxy1 = " ( - daughter(1).vx * daughter(1).py + daughter(1).vy * daughter(1).px) / daughter(1).pt "
## # dz constructed from vertex position
## dz0 = "  daughter(0).vz -  ( daughter(0).vx * daughter(0).px  + daughter(0).vy * daughter(0).py) / daughter(0).pt *  daughter(0).pz / daughter(0).pt "
## dz1 = " daughter(1).vz -  ( daughter(1).vx * daughter(1).px  + daughter(1).vy * daughter(1).py) / daughter(1).pt *  daughter(1).pz / daughter(1).pt "


#ZMuMu at least 1 HLT + 2 track-iso (Shape)
goodZToMuMuPlotsTemplate = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLT")
)

#ZMuMu at least 1 HLT + at least 1 NON track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")
setattr(process, "nonIsolatedZToMuMuPlots", nonIsolatedZToMuMuPlots)

#ZMuMu at least 1 HLT + 1 NON track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
oneNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
oneNonIsolatedZToMuMuPlots.src = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
setattr(process, "oneNonIsolatedZToMuMuPlots", oneNonIsolatedZToMuMuPlots)

#ZMuMu at least 1 HLT + 2 NON track-iso
process.goodZToMuMuPlots = goodZToMuMuPlotsTemplate
twoNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
twoNonIsolatedZToMuMuPlots.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
setattr(process, "twoNonIsolatedZToMuMuPlots", twoNonIsolatedZToMuMuPlots)




etaBounds = [2.1]
## if you want to perform studies on different eta bins...
##### etaBounds = [-2.1, -1.2, -0.8, 0.8, 1.2, 2.1]

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

#ZMuMu at least  1 HLT + 2  track-iso
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


#ZMuMu same charge
process.goodZToMuMuSameChargeAtLeast1HLT = copy.deepcopy(process.goodZToMuMuAtLeast1HLT)
process.goodZToMuMuSameChargeAtLeast1HLT.src= cms.InputTag("goodZToMuMuSameCharge")

process.goodZToMuMuSameChargeAtLeast1HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuSameChargeAtLeast1HLTPlots.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")
process.goodZToMuMuSameCharge2HLT = copy.deepcopy(process.goodZToMuMu2HLT)
process.goodZToMuMuSameCharge2HLT.src= cms.InputTag("goodZToMuMuSameCharge")

process.goodZToMuMuSameCharge2HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuSameCharge2HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge2HLT")

process.goodZToMuMuSameCharge1HLT = copy.deepcopy(process.goodZToMuMu1HLT)
process.goodZToMuMuSameCharge1HLT.src= cms.InputTag("goodZToMuMuSameCharge")

process.goodZToMuMuSameCharge1HLTPlots = copy.deepcopy(goodZToMuMuPlotsTemplate)
process.goodZToMuMuSameCharge1HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge1HLT")


## process.globalMuQualityCutsAnalysis= cms.EDAnalyzer(
##     "GlbMuQualityCutsAnalysis",
##     src = cms.InputTag("goodZToMuMuAtLeast1HLT"), 
##     ptMin = cms.untracked.double(0.0),
##     massMin = cms.untracked.double(0.0),
##     massMax = cms.untracked.double(120.0),
##     etaMin = cms.untracked.double(-1.0),
##     etaMax = cms.untracked.double(10.0),
##     trkIso = cms.untracked.double(10000),
##     chi2Cut = cms.untracked.double(10),
##     nHitCut = cms.untracked.int32(10)
##  )


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

process.goodZToMuMu2HLTVtxed = cms.EDProducer(
    "KalmanVertexFitCompositeCandProducer",
    src = cms.InputTag("goodZToMuMu2HLT")
)

process.goodZToMuMu2HLTVtxedNtuple = cms.EDProducer(
    "CandViewNtpProducer",
    src = cms.InputTag("goodZToMuMu2HLTVtxed"),
    variables = cms.VPSet(
      cms.PSet(
        tag = cms.untracked.string("mass"),
        quantity = cms.untracked.string("mass")
      ),
       cms.PSet(
        tag = cms.untracked.string("vertexNdof"),
        quantity = cms.untracked.string("vertexNdof")
      ),
       cms.PSet(
        tag = cms.untracked.string("vertexNormalizedChi2"),
        quantity = cms.untracked.string("vertexNormalizedChi2")
      ),
    )
)







process.goodZToMuMuEdmNtuple = cms.EDProducer(
    "ZToLLEdmNtupleDumper",
    zBlocks = cms.VPSet(
    cms.PSet(
        zName = cms.string("zGolden"),
        z = cms.InputTag("goodZToMuMuAtLeast1HLT"),
        zGenParticlesMatch = cms.InputTag(""),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"), 
        ptThreshold = cms.double(1.5),
        etEcalThreshold = cms.double(0.2),
        etHcalThreshold = cms.double(0.5),
        deltaRVetoTrk = cms.double(0.015),
        deltaRTrk = cms.double(0.3),
        deltaREcal = cms.double(0.25),
        deltaRHcal = cms.double(0.25),
        alpha = cms.double(0.),
        beta = cms.double(-0.75),
        relativeIsolation = cms.bool(False),
        hltPath = cms.string("HLT_Mu9")
        
      ),
     )
)



process.goodZToMuMu2HLTEdmNtuple = copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.goodZToMuMu2HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMu2HLT")
process.goodZToMuMu2HLTEdmNtuple.zBlocks[0].zName = cms.string("zGolden2HLT")


process.goodZToMuMu1HLTEdmNtuple = copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.goodZToMuMu1HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMu1HLT")
process.goodZToMuMu1HLTEdmNtuple.zBlocks[0].zName = cms.string("zGolden1HLT")

process.oneNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
process.oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z1NotIso")

process.twoNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
process.twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z2NotIso")


process.goodZToMuMuSameChargeEdmNtuple= copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.goodZToMuMuSameChargeEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")
process.goodZToMuMuSameChargeEdmNtuple.zBlocks[0].zName = cms.string("zSameCharge")


process.goodZToMuMuOneStandAloneEdmNtuple= copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.goodZToMuMuOneStandAloneEdmNtuple.zBlocks[0].z=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")
process.goodZToMuMuOneStandAloneEdmNtuple.zBlocks[0].zName=cms.string("zMuSta")

process.goodZToMuMuOneTrackEdmNtuple= copy.deepcopy(process.goodZToMuMuEdmNtuple)
process.goodZToMuMuOneTrackEdmNtuple.zBlocks[0].z=cms.InputTag("goodZToMuMuOneTrackFirstHLT")
process.goodZToMuMuOneTrackEdmNtuple.zBlocks[0].zName=cms.string("zMuTrk")




process.initialGoodZToMuMuPath = cms.Path( 
    process.goodZToMuMu +
    process.zmumuSaMassHistogram     
)

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMuAtLeast1HLT+
    process.goodZToMuMuPlots ,
#    process.goodZToMuMuEdmNtuple,
   # process.globalMuQualityCutsAnalysis,
    "goodZToMuMu", "goodZToMuMu",
    "double")


addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMu2HLT +
    process.goodZToMuMu2HLTPlots +
    process.goodZToMuMu2HLTEdmNtuple +
    process.goodZToMuMu2HLTVtxed +
    process.goodZToMuMu2HLTVtxedNtuple,
    "goodZToMuMu2HLT", "goodZToMuMu2HLT",
    "double")

addModulesFromTemplate(
    process.goodZToMuMu +
    process.goodZToMuMu1HLT +
    process.goodZToMuMu1HLTPlots +
    process.goodZToMuMu1HLTEdmNtuple,
    "goodZToMuMu1HLT", "goodZToMuMu1HLT",
    "double")

#process.globalMuQualityCutsAnalysisSameCharge = copy.deepcopy(process.globalMuQualityCutsAnalysis)
#process.globalMuQualityCutsAnalysisSameCharge.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")

addModulesFromTemplate(
    process.dimuonsGlobalSameCharge+
    process.goodZToMuMuSameCharge +
    process.goodZToMuMuSameChargeAtLeast1HLT+
    process.goodZToMuMuSameChargeAtLeast1HLTPlots +
    process.goodZToMuMuSameChargeEdmNtuple, 
#    process.globalMuQualityCutsAnalysisSameCharge,
    "goodZToMuMuSameCharge", "goodZToMuMuSameCharge",
    "double")


addModulesFromTemplate(
    process.dimuonsGlobalSameCharge+
    process.goodZToMuMuSameCharge +
    process.goodZToMuMuSameCharge2HLT +
    process.goodZToMuMuSameCharge2HLTPlots,
    "goodZToMuMuSameCharge2HLT", "goodZToMuMuSameCharge2HLT",
    "double")

addModulesFromTemplate(
    process.dimuonsGlobalSameCharge+
    process.goodZToMuMuSameCharge +
    process.goodZToMuMuSameCharge1HLT +
    process.goodZToMuMuSameCharge1HLTPlots,
    "goodZToMuMuSameCharge1HLT", "goodZToMuMuSameCharge1HLT",
    "double")



    
process.nonIsolatedZToMuMuPath = cms.Path (
    process.nonIsolatedZToMuMu +
    process.nonIsolatedZToMuMuAtLeast1HLT +
    process.nonIsolatedZToMuMuPlots 
 #   process.nonIsolatedZToMuMuEdmNtuple
)
process.oneNonIsolatedZToMuMuPath = cms.Path(
    process.nonIsolatedZToMuMu +
    process.oneNonIsolatedZToMuMu +
    process.oneNonIsolatedZToMuMuAtLeast1HLT +
    process.oneNonIsolatedZToMuMuPlots+
    process.oneNonIsolatedZToMuMuEdmNtuple
)

process.twoNonIsolatedZToMuMuPath = cms.Path(
    process.nonIsolatedZToMuMu +
    process.twoNonIsolatedZToMuMu +
    process.twoNonIsolatedZToMuMuAtLeast1HLT +
    process.twoNonIsolatedZToMuMuPlots +
    process.twoNonIsolatedZToMuMuEdmNtuple
    
)

addModulesFromTemplate(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.goodZToMuMuOneStandAloneMuonNtuple +
    process.goodZToMuMuOneStandAloneEdmNtuple +
    process.goodZToMuMuOneStandAloneMuonPlots, 
    "goodZToMuMuOneStandAloneMuon", "goodZToMuMuOneStandAloneMuon",
    "single")

addModulesFromTemplate(
    ~process.goodZToMuMu +
    ~process.zToMuMuOneStandAloneMuon +
    process.zToMuGlobalMuOneTrack +
    process.zToMuMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackFirstHLT +
    process.goodZToMuMuOneTrackPlots +
    process.goodZToMuMuOneTrackEdmNtuple ,
    "goodZToMuMuOneTrack", "goodZToMuMuOneTrack",
    "single")

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('Ntuple_test_v2.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMu1HLTEdmNtuple_*_*",
      "keep *_goodZToMuMu2HLTEdmNtuple_*_*",
      "keep *_goodZToMuMuSameChargeEdmNtuple_*_*",
      "keep *_nonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_oneNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_twoNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMuOneStandAloneEdmNtuple_*_*",
      "keep *_goodZToMuMuOneTrackEdmNtuple_*_*",
      "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuOneStandAloneMuonPath",
        "goodZToMuMuPath",
        "goodZToMuMu1HLTPath",
        "goodZToMuMu2HLTPath",
        "goodZToMuMuSameChargePath",
        "nonIsolatedZToMuMuPath",
        "oneNonIsolatedZToMuMuPath",
        "twoNonIsolatedZToMuMuPath",
        "goodZToMuMuOneTrackPath",
      )
    )
)
  
process.endPath = cms.EndPath( 
    process.eventInfo
   + process.out
)

