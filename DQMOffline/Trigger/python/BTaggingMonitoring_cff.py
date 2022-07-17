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
    bjetSelection = 'pt>5 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4DiJet20_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,10,15,20,30,50,70,100,150,200,400,700,1000,1500,3000])
)

BTagMu_AK4DiJet40_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet40_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4DiJet40_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,30,40,50,70,100,150,200,400,700,1000,1500,3000])
)

BTagMu_AK4DiJet70_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet70_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>50 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4DiJet70_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,50,60,70,80,90,100,150,200,400,700,1000,1500,3000])
)

BTagMu_AK4DiJet110_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet110_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>90 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4DiJet110_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,90,100,110,120,130,150,200,400,700,1000,1500,3000])
)


BTagMu_AK4DiJet170_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet170_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4DiJet170_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,150,160,170,180,190,200,400,700,1000,1500,3000])
)


BTagMu_AK4Jet300_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_Jet/BTagMu_AK4Jet300_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 1,
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>250 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK4Jet300_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500,3000])
)


#BTagMu AK8
BTagMu_AK8DiJet170_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_DiJet/BTagMu_AK8DiJet170_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 2,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK8DiJet170_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,150,160,170,180,190,200,400,700,1000,1500,3000])
)


BTagMu_AK8Jet300_Mu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagMu_Jet/BTagMu_AK8Jet300_Mu5',
    nmuons = 1,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>250 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK8Jet300_Mu5_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500,3000])
)


BTagMu_AK8Jet170_DoubleMu5 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/BTagDiMu_Jet/BTagMu_AK8Jet170_DoubleMu5',
    nmuons = 2,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    muoSelection = 'pt>7 & abs(eta)<2.4 & isPFMuon & isGlobalMuon & innerTrack.hitPattern.numberOfValidTrackerHits>7 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & numberOfMatchedStations>1 &globalTrack.normalizedChi2<10',
    jetSelection = 'pt>150 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_BTagMu_AK8Jet170_DoubleMu5_v*']),
    histoPSet = dict(jetPtBinning = [0,150,160,170,180,190,200,400,700,1000,1500,3000])
)

# PFJet AK4
PFJet40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJet40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4',
    histoPSet = dict(jetPtBinning = [0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet40_v*'])
)

# PFJet AK8
AK8PFJet40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJet40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>30 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_AK8PFJet40_v*']),
    histoPSet = dict(jetPtBinning = [0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000])
)

# PFJetFwd AK4
PFJetFwd40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/PFJetFwd40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jetSelection = 'pt>30 & abs(eta)>2.7 & abs(eta)<5.0',
    bjetSelection = 'pt>20 & abs(eta)>2.7 & abs(eta)<5.0',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJetFwd40_v*']),
    histoPSet = dict(
        jetPtBinning = [0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000],
        jetEtaBinning = [-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0],
        jetEtaBinning2D = [-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0],
        etaPSet = dict(nbins=50, xmin=-5.0, xmax=5.0)
    )
)

# PFJetFwd AK8
AK8PFJetFwd40 = hltBTVmonitoring.clone(
    FolderName = 'HLT/BTV/PFJet/AK8PFJetFwd40',
    nmuons = 0,
    nelectrons = 0,
    njets = 1,
    jets = "ak8PFJetsPuppi",
    jetSelection = 'pt>30 & abs(eta)>2.7 & abs(eta)<5.0',
    bjetSelection = 'pt>20 & abs(eta)>2.7 & abs(eta)<5.0',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_AK8PFJetFwd40_v*']),
    histoPSet = dict(
        jetPtBinning = [0,30,35,40,45,50,60,70,100,150,200,400,700,1000,1500,3000],
        jetEtaBinning = [-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0],
        jetEtaBinning2D = [-5.0,-4.7,-4.4,-4.1,-3.8,-3.5,-3.2,-2.9,-2.7,-2.4,-2.1,0.0,2.1,2.4,2.7,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5.0],
        etaPSet = dict(nbins=50, xmin=-5.0, xmax=5.0)
    )
)

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
  + AK8PFJet40
  + PFJetFwd40
  + AK8PFJetFwd40
)
