import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitoring_cfi import mssmHbbMonitoring

MUON_PT_BINNING = [0,4,6,7,8,9,10,11,12,13,14,15,20,30,40,50,100,200]

#Define MssmHbb specific cuts 
hltMssmHbbmonitoring =  mssmHbbMonitoring.clone(
    btagAlgos = ["pfParticleNetAK4DiscriminatorsJetTagsForRECO:BvsAll"],
    workingpoint    = 0.1919, # medium WP
    njets = 2,
    nbjets = 2,
    nmuons = 0,
    bJetDeltaEtaMax = 1.6, # deta cut between leading bjets
    bJetMuDeltaRmax = 0.4  # dR(mu,nbjet) cone; only if #mu >1
)
# Fully-hadronic MssmHbb  (main)
hltMssmHbbMonitoringFH116 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/fullhadronic/HLT_DoublePFJets116MaxDeta1p6_PNet2BTag_0p11',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets116MaxDeta1p6_PNet2BTag_0p11_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Fully-hadronic MssmHbb  (backup)
hltMssmHbbMonitoringFH128 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/fullhadronic/HLT_DoublePFJets128MaxDeta1p6_PNet2BTag_0p11',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets128MaxDeta1p6_PNet2BTag_0p11_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Semileptonic MssmHbb  (main)
hltMssmHbbMonitoringSL40 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets40MaxDeta1p6_PNet2BTag_0p11',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40MaxDeta1p6_PNet2BTag_0p11_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = MUON_PT_BINNING)

)

# Semileptonic MssmHbb (backup)
hltMssmHbbMonitoringSL54 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets54MaxDeta1p6_PNet2BTag_0p11',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets54MaxDeta1p6_PNet2BTag_0p11_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = MUON_PT_BINNING)
)


#control b-tagging 
hltMssmHbbMonitoringMu12 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/control/muon/HLT_Mu12eta2p3',
    nmuons = 1,
    nbjets = 0,
    njets = 0,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12eta2p3_v*']),
    histoPSet = dict(muPtBinning = MUON_PT_BINNING)
)

hltMssmHbbMonitoringMu12Jet40 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/SUS/MssmHbb/control/muon/HLT_Mu12eta2p3_PFJet40',
    nmuons = 1,
    nbjets = 0,
    njets = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>40 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12eta2p3_PFJet40_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = MUON_PT_BINNING)
)



mssmHbbMonitorHLT = cms.Sequence(
    hltMssmHbbMonitoringFH116 +
    hltMssmHbbMonitoringFH128 +
    hltMssmHbbMonitoringSL40  +  
    hltMssmHbbMonitoringSL54  +  
    hltMssmHbbMonitoringMu12 +
    hltMssmHbbMonitoringMu12Jet40
)
