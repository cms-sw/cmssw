import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitoring_cfi import mssmHbbMonitoring

#Define MssmHbb specific cuts 
hltMssmHbbmonitoring =  mssmHbbMonitoring.clone(
    btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"],
    workingpoint    = 0.92, # tight WP
    bJetDeltaEtaMax = 1.6,   # deta cut between leading bjets
    bJetMuDeltaRmax = 0.4   # dR(mu,nbjet) cone; only if #mu >1
)
# Fully-hadronic MssmHbb
hltMssmHbbmonitoringAL100 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/fullhadronic/pt100'
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/pt100',
    nmuons = 0,
    nbjets = 2,
    bjetSelection = 'pt>110 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)


hltMssmHbbmonitoringAL116 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/fullhadronic/pt116',
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/pt116',
    nmuons = 0,
    nbjets = 2,
    bjetSelection = 'pt>116 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)


hltMssmHbbmonitoringAL128 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/fullhadronic/pt128',
    FolderName = 'HLT/HIG/MssmHbb/fullhadronic/pt128',
    nmuons = 0,
    nbjets = 2,
    bjetSelection = 'pt>128 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)


# Semi-leptonic MssmHbb(mu)
hltMssmHbbmonitoringSL40 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/semileptonic/pt40',
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/pt40',
    nmuons = 1,
    nbjets = 2,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    bjetSelection = 'pt>40 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets40MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)

hltMssmHbbmonitoringSL54 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/semileptonic/pt54',
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/pt54',
    nmuons = 1,
    nbjets = 2,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    bjetSelection = 'pt>54 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets54MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)


hltMssmHbbmonitoringSL62 = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/semileptonic/pt62'
    FolderName = 'HLT/HIG/MssmHbb/semileptonic/pt62',
    nmuons = 1,
    nbjets = 2,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    bjetSelection = 'pt>62 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets62MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)


#control b-tagging 
hltMssmHbbmonitoringControl = hltMssmHbbmonitoring.clone(
    #FolderName = 'HLT/Higgs/MssmHbb/control/mu12_pt30_nobtag',
    FolderName = 'HLT/HIG/MssmHbb/control/mu12_pt30_nobtag',
    nmuons = 1,
    nbjets = 0,
    njets = 1,
    muoSelection = 'pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10',
    jetSelection = 'pt>40 & abs(eta)<2.2',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_SingleJet30_Mu12_SinglePFJet40_v*']),
    histoPSet = dict(jetPtBinning = [0,250,280,300,320,360,400,700,1000,1500])
)



mssmHbbMonitorHLT = cms.Sequence(
    #full-hadronic
    hltMssmHbbmonitoringAL100
    + hltMssmHbbmonitoringAL116
    + hltMssmHbbmonitoringAL128
    #semileptonic
    + hltMssmHbbmonitoringSL40
    + hltMssmHbbmonitoringSL54
    + hltMssmHbbmonitoringSL62    
    #control no b-tag
    + hltMssmHbbmonitoringControl
)
