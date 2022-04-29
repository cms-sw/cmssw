import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitoring_cfi import mssmHbbMonitoring

#Define MssmHbb specific cuts 
hltMssmHbbmonitoring =  mssmHbbMonitoring.clone(
    btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint    = 0.2783, # medium WP
    njets = 2,
    nbjets = 2,
    nmuons = 0,
    bJetDeltaEtaMax = 1.6, # deta cut between leading bjets
    bJetMuDeltaRmax = 0.4  # dR(mu,nbjet) cone; only if #mu >1
)
# Fully-hadronic MssmHbb DeepCSV (main)
hltMssmHbbDeepCSVMonitoringFH116 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepCSV_p71',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Fully-hadronic MssmHbb DeepCSV (backup)
hltMssmHbbDeepCSVMonitoringFH128 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepCSV_p71',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Semileptonic MssmHbb DeepCSV (main)
hltMssmHbbDeepCSVMonitoringSL40 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])

)

# Semileptonic MssmHbb DeepCSV (backup)
hltMssmHbbDeepCSVMonitoringSL54 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepCSV_p71',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])
)




# Fully-hadronic MssmHbb DeepJet (main)
hltMssmHbbDeepJetMonitoringFH116 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepJet_p71',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepJet_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Fully-hadronic MssmHbb DeepJet (backup)
hltMssmHbbDeepJetMonitoringFH128 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepJet_p71',
    bjetSelection = 'pt>100 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepJet_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,100,150,200,250,300,350,400,500,700,1000,1500])
)

# Semileptonic MssmHbb DeepJet (main)
hltMssmHbbDeepJetMonitoringSL40 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])
)

# Semileptonic MssmHbb DeepJet (backup)
hltMssmHbbDeepJetMonitoringSL54 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepJet_p71',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    nmuons = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepJet_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])
)



#control b-tagging 
hltMssmHbbMonitoringMu12 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/control/muon/HLT_Mu12eta2p3',
    nmuons = 1,
    nbjets = 0,
    njets = 0,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12eta2p3_v*']),
    histoPSet = dict(muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])
)

hltMssmHbbMonitoringMu12Jet40 = hltMssmHbbmonitoring.clone(
    FolderName = 'HLT/HIG/MssmHbb/control/muon/HLT_Mu12eta2p3_PFJet40',
    nmuons = 1,
    nbjets = 0,
    njets = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>40 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12eta2p3_PFJet40_v*']),
    histoPSet = dict(jetPtBinning = [0,40,60,80,120,160,250,300,350,400,500,1000,1500],
                     muPtBinning = [0,7,11,12,13,15,20,30,40,50,70,100,150,200,400,700])
)



mssmHbbMonitorHLT = cms.Sequence(
    #full-hadronic DeepCSV
    hltMssmHbbDeepCSVMonitoringFH116 +
    hltMssmHbbDeepCSVMonitoringFH128 +
    #semileptonic DeepCSV
    hltMssmHbbDeepCSVMonitoringSL40  +  
    hltMssmHbbDeepCSVMonitoringSL54  +  
    #full-hadronic DeepJet
    hltMssmHbbDeepJetMonitoringFH116 +
    hltMssmHbbDeepJetMonitoringFH128 +
    #semileptonic DeepJet
    hltMssmHbbDeepJetMonitoringSL40  +  
    hltMssmHbbDeepJetMonitoringSL54  +  
    #muon jet no b-tag
    hltMssmHbbMonitoringMu12 +
    hltMssmHbbMonitoringMu12Jet40
)
