import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTaggingMonitor_cfi import hltBTVmonitoring

# BTagMu AK4
BTagMu_AK4DiJet20_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet20_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>10 & abs(eta)<2.4',
    bjetSelection = 'pt>5 & abs(eta)<2.4'
)
BTagMu_AK4DiJet20_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet20_Mu5_v*')
BTagMu_AK4DiJet20_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,10,15,20,30,50,70,100,150,200,400,700,1000,1500,3000)

BTagMu_AK4DiJet40_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet40_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4'
)
BTagMu_AK4DiJet40_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet40_Mu5_v*')
BTagMu_AK4DiJet40_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,70,100,150,200,400,700,1000,1500,3000)

BTagMu_AK4DiJet70_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet70_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>50 & abs(eta)<2.4'
)
BTagMu_AK4DiJet70_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet70_Mu5_v*')
BTagMu_AK4DiJet70_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,50,60,70,80,90,100,150,200,400,700,1000,1500,3000)

BTagMu_AK4DiJet110_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet110_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>90 & abs(eta)<2.4'
)
BTagMu_AK4DiJet110_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet110_Mu5_v*')
BTagMu_AK4DiJet110_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,90,100,110,120,130,150,200,400,700,1000,1500,3000)

BTagMu_AK4DiJet170_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet170_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4'
)
BTagMu_AK4DiJet170_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet170_Mu5_v*')
BTagMu_AK4DiJet170_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,150,160,170,180,190,200,400,700,1000,1500,3000)

BTagMu_AK4Jet300_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_Jet/BTagMu_AK4Jet300_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 1,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>250 & abs(eta)<2.4'
)
BTagMu_AK4Jet300_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4Jet300_Mu5_v*')
BTagMu_AK4Jet300_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500,3000)

#BTagMu AK8
BTagMu_AK8DiJet170_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK8DiJet170_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4'
)
BTagMu_AK8DiJet170_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK8DiJet170_Mu5_v*')
BTagMu_AK8DiJet170_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,150,160,170,180,190,200,400,700,1000,1500,3000)

BTagMu_AK8Jet300_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_Jet/BTagMu_AK8Jet300_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>250 & abs(eta)<2.4'
    
)
BTagMu_AK8Jet300_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK8Jet300_Mu5_v*')
BTagMu_AK8Jet300_Mu5.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500,3000)

BTagMu_AK8Jet170_DoubleMu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagDiMu_Jet/BTagMu_AK8Jet170_DoubleMu5',
    nmuons = 2,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>7 & abs(eta)<2.4 & isPFMuon & isGlobalMuon & innerTrack.hitPattern.numberOfValidTrackerHits>7 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & numberOfMatchedStations>1 &globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4'
)
BTagMu_AK8Jet170_DoubleMu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK8Jet170_DoubleMu5_v*')
BTagMu_AK8Jet170_DoubleMu5.histoPSet.jetPtBinning = cms.vdouble(0,150,160,170,180,190,200,400,700,1000,1500,3000)

# PFJet AK4
PFJet40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4'
)
PFJet40.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000)
PFJet40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet40_v*')

PFJet60 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet60',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>50 & abs(eta)<2.4'
)
PFJet60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet60_v*')
PFJet60.histoPSet.jetPtBinning = cms.vdouble(0,50,55,60,65,70,80,90,100,120,150,200,400,700,1000,1500,3000)

PFJet80 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet80',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>70 & abs(eta)<2.4'
)
PFJet80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet80_v*')
PFJet80.histoPSet.jetPtBinning = cms.vdouble(0,70,75,80,85,90,100,120,150,200,400,700,1000,1500,3000)

PFJet140 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet140',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>120 & abs(eta)<2.4'
)
PFJet140.histoPSet.jetPtBinning = cms.vdouble(0,120,130,140,150,160,170,200,400,700,1000,1500,3000)
PFJet140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet140_v*')

PFJet200 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet200',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>170 & abs(eta)<2.4'
)
PFJet200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet200_v*')
PFJet200.histoPSet.jetPtBinning = cms.vdouble(0,170,180,190,200,210,220,250,300,400,700,1000,1500,3000)

PFJet260 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet260',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>220 & abs(eta)<2.4'
)
PFJet260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet260_v*')
PFJet260.histoPSet.jetPtBinning = cms.vdouble(0,220,240,260,280,300,350,400,700,1000,1500,3000)

PFJet320 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet320',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>280 & abs(eta)<2.4'
)
PFJet320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet320_v*')
PFJet320.histoPSet.jetPtBinning = cms.vdouble(0,280,300,320,340,360,400,500,700,1000,1500,3000)

PFJet400 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet400',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>350 & abs(eta)<2.4'
)
PFJet400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet400_v*')
PFJet400.histoPSet.jetPtBinning = cms.vdouble(0,350,380,400,420,450,500,700,1000,1500,3000)

PFJet450 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet450',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>400 & abs(eta)<2.4'
)
PFJet450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet450_v*')
PFJet450.histoPSet.jetPtBinning = cms.vdouble(0,400,430,450,470,500,700,1000,1500,3000)

PFJet500 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet500',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>450 & abs(eta)<2.4'
)
PFJet500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet500_v*')
PFJet500.histoPSet.jetPtBinning = cms.vdouble(0,450,480,500,520,550,600,700,1000,1500,3000)

PFJet550 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet550',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>500 & abs(eta)<2.4'
)
PFJet550.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet550_v*')
PFJet550.histoPSet.jetPtBinning = cms.vdouble(0,500,520,550,600,700,1000,1500,3000)
# PFJet AK8
AK8PFJet40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4'
)
AK8PFJet40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet40_v*')
AK8PFJet40.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000)

AK8PFJet60 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet60',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>50 & abs(eta)<2.4'
)
AK8PFJet60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet60_v*')
AK8PFJet60.histoPSet.jetPtBinning = cms.vdouble(0,50,55,60,65,70,80,90,100,120,150,200,400,700,1000,1500,3000)

AK8PFJet80 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet80',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>70 & abs(eta)<2.4'
)
AK8PFJet80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet80_v*')
AK8PFJet80.histoPSet.jetPtBinning = cms.vdouble(0,70,75,80,85,90,100,120,150,200,400,700,1000,1500,3000)

AK8PFJet140 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet140',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>120 & abs(eta)<2.4'
)
AK8PFJet140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet140_v*')
AK8PFJet140.histoPSet.jetPtBinning = cms.vdouble(0,120,130,140,150,160,170,200,400,700,1000,1500,3000)

AK8PFJet200 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet200',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>170 & abs(eta)<2.4'
)
AK8PFJet200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet200_v*')
AK8PFJet200.histoPSet.jetPtBinning = cms.vdouble(0,170,180,190,200,210,220,250,300,400,700,1000,1500,3000)

AK8PFJet260 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet260',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>220 & abs(eta)<2.4'
)
AK8PFJet260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet260_v*')
AK8PFJet260.histoPSet.jetPtBinning = cms.vdouble(0,220,240,260,280,300,350,400,700,1000,1500,3000)

AK8PFJet320 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet320',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>280 & abs(eta)<2.4'
)
AK8PFJet320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet320_v*')
AK8PFJet320.histoPSet.jetPtBinning = cms.vdouble(0,280,300,320,340,360,400,500,700,1000,1500,3000)

AK8PFJet400 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet400',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>350 & abs(eta)<2.4'
)
AK8PFJet400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet400_v*')
AK8PFJet400.histoPSet.jetPtBinning = cms.vdouble(0,350,380,400,420,450,500,700,1000,1500,3000)

AK8PFJet450 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet450',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>400 & abs(eta)<2.4'
)
AK8PFJet450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet450_v*')
AK8PFJet450.histoPSet.jetPtBinning = cms.vdouble(0,400,430,450,470,500,700,1000,1500,3000)

AK8PFJet500 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet500',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>450 & abs(eta)<2.4'
)
AK8PFJet500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet500_v*')
AK8PFJet500.histoPSet.jetPtBinning = cms.vdouble(0,450,480,500,520,550,600,700,1000,1500,3000)

AK8PFJet550 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet550',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>500 & abs(eta)<2.4'
)
AK8PFJet550.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet550_v*')
AK8PFJet550.histoPSet.jetPtBinning = cms.vdouble(0,500,520,550,600,700,1000,1500,3000)

# PFJetFwd AK4
PFJetFwd40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>30 & abs(eta)>2.7 & abs(eta)<5.0',
    bjetSelection = 'pt>20 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd40_v*')
PFJetFwd40.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000)
PFJetFwd40.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd40.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd40.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd60 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd60',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>50 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd60_v*')
PFJetFwd60.histoPSet.jetPtBinning = cms.vdouble(0,50,55,60,65,70,80,90,100,120,150,200,400,700,1000,1500,3000)
PFJetFwd60.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd60.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd60.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd80 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd80',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>70 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd80_v*')
PFJetFwd80.histoPSet.jetPtBinning = cms.vdouble(0,70,75,80,85,90,100,120,150,200,400,700,1000,1500,3000)
PFJetFwd80.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd80.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd80.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd140 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd140',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>120 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd140_v*')
PFJetFwd140.histoPSet.jetPtBinning = cms.vdouble(0,120,130,140,150,160,170,200,400,700,1000,1500,3000)
PFJetFwd140.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd140.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd140.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd200 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd200',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>170 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd200_v*')
PFJetFwd200.histoPSet.jetPtBinning = cms.vdouble(0,170,180,190,200,210,220,250,300,400,700,1000,1500,3000)
PFJetFwd200.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd200.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd200.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd260 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd260',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>220 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd260_v*')
PFJetFwd260.histoPSet.jetPtBinning = cms.vdouble(0,220,240,260,280,300,350,400,700,1000,1500,3000)
PFJetFwd260.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd260.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd260.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd320 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd320',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>280 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd320_v*')
PFJetFwd320.histoPSet.jetPtBinning = cms.vdouble(0,280,300,320,340,360,400,500,700,1000,1500,3000)
PFJetFwd320.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd320.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd320.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd400 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd400',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>350 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd400_v*')
PFJetFwd400.histoPSet.jetPtBinning = cms.vdouble(0,350,380,400,420,450,500,700,1000,1500,3000)
PFJetFwd400.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd400.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd400.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd450 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd450',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>400 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd450_v*')
PFJetFwd450.histoPSet.jetPtBinning = cms.vdouble(0,400,430,450,470,500,700,1000,1500,3000)
PFJetFwd450.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd450.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd450.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

PFJetFwd500 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd500',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>450 & abs(eta)>2.7 & abs(eta)<5.0'
)
PFJetFwd500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJetFwd500_v*')
PFJetFwd500.histoPSet.jetPtBinning = cms.vdouble(0,450,480,500,520,550,600,700,1000,1500,3000)
PFJetFwd500.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd500.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
PFJetFwd500.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))
# PFJetFwd AK8
AK8PFJetFwd40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>30 & abs(eta)>2.7 & abs(eta)<5.0',
    bjetSelection = 'pt>20 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd40_v*')
AK8PFJetFwd40.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000)
AK8PFJetFwd40.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd40.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd40.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd60 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd60',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>50 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd60_v*')
AK8PFJetFwd60.histoPSet.jetPtBinning = cms.vdouble(0,50,55,60,65,70,80,90,100,120,150,200,400,700,1000,1500,3000)
AK8PFJetFwd60.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd60.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd60.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd80 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd80',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>70 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd80_v*')
AK8PFJetFwd80.histoPSet.jetPtBinning = cms.vdouble(0,70,75,80,85,90,100,120,150,200,400,700,1000,1500,3000)
AK8PFJetFwd80.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd80.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd80.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd140 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd140',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>120 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd140_v*')
AK8PFJetFwd140.histoPSet.jetPtBinning = cms.vdouble(0,120,130,140,150,160,170,200,400,700,1000,1500,3000)
AK8PFJetFwd140.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd140.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd140.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd200 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd200',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>170 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd200_v*')
AK8PFJetFwd200.histoPSet.jetPtBinning = cms.vdouble(0,170,180,190,200,210,220,250,300,400,700,1000,1500,3000)
AK8PFJetFwd200.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd200.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd200.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd260 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd260',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>220 & abs(eta)>2.7 & abs(eta)<5.0'

)
AK8PFJetFwd260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd260_v*')
AK8PFJetFwd260.histoPSet.jetPtBinning = cms.vdouble(0,220,240,260,280,300,350,400,700,1000,1500,3000)
AK8PFJetFwd260.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd260.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd260.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd320 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd320',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>280 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd320_v*')
AK8PFJetFwd320.histoPSet.jetPtBinning = cms.vdouble(0,280,300,320,340,360,400,500,700,1000,1500,3000)
AK8PFJetFwd320.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd320.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd320.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd400 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd400',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>350 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd400_v*')
AK8PFJetFwd400.histoPSet.jetPtBinning = cms.vdouble(0,350,380,400,420,450,500,700,1000,1500,3000)
AK8PFJetFwd400.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd400.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd400.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd450 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd450',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>400 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd450_v*')
AK8PFJetFwd450.histoPSet.jetPtBinning = cms.vdouble(0,400,430,450,470,500,700,1000,1500,3000)
AK8PFJetFwd450.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd450.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd450.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

AK8PFJetFwd500 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd500',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>450 & abs(eta)>2.7 & abs(eta)<5.0'
)
AK8PFJetFwd500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJetFwd500_v*')
AK8PFJetFwd500.histoPSet.jetPtBinning = cms.vdouble(0,450,480,500,520,550,600,700,1000,1500,3000)
AK8PFJetFwd500.histoPSet.jetEtaBinning = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd500.histoPSet.jetEtaBinning2D = cms.vdouble(-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0)
AK8PFJetFwd500.histoPSet.etaPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(-5.0), xmax=cms.double(5.0))

### Sequences

btagMonitorHLT = cms.Sequence(
    BTagMu_AK4DiJet20_Mu5
  + BTagMu_AK4DiJet40_Mu5
  + BTagMu_AK4DiJet70_Mu5
  + BTagMu_AK4DiJet110_Mu5
  + BTagMu_AK4DiJet170_Mu5
  + BTagMu_AK8DiJet170_Mu5
  + BTagMu_AK8Jet170_DoubleMu5
  + BTagMu_AK4Jet300_Mu5
  + BTagMu_AK8Jet300_Mu5
)

btvHLTDQMSourceExtra = cms.Sequence(
    PFJet40
  + PFJet60
  + PFJet80
  + PFJet140
  + PFJet200
  + PFJet260
  + PFJet320
  + PFJet400
  + PFJet450
  + PFJet500
  + PFJet550
  + AK8PFJet40
  + AK8PFJet60
  + AK8PFJet80
  + AK8PFJet140
  + AK8PFJet200
  + AK8PFJet260
  + AK8PFJet320
  + AK8PFJet400
  + AK8PFJet450
  + AK8PFJet500
  + AK8PFJet550
  + PFJetFwd40
  + PFJetFwd60
  + PFJetFwd80
  + PFJetFwd140
  + PFJetFwd200
  + PFJetFwd260
  + PFJetFwd320
  + PFJetFwd400
  + PFJetFwd450
  + PFJetFwd500
  + AK8PFJetFwd40
  + AK8PFJetFwd60
  + AK8PFJetFwd80
  + AK8PFJetFwd140
  + AK8PFJetFwd200
  + AK8PFJetFwd260
  + AK8PFJetFwd320
  + AK8PFJetFwd400
  + AK8PFJetFwd450
  + AK8PFJetFwd500
)
