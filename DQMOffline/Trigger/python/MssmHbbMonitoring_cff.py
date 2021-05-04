import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitoring_cfi import mssmHbbMonitoring

#Define MssmHbb specific cuts 
hltMssmHbbmonitoring =  mssmHbbMonitoring.clone()
hltMssmHbbmonitoring.btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"]
hltMssmHbbmonitoring.workingpoint    = cms.double(0.92) # tight WP
hltMssmHbbmonitoring.bJetDeltaEtaMax = cms.double(1.6)   # deta cut between leading bjets
hltMssmHbbmonitoring.bJetMuDeltaRmax = cms.double(0.4)   # dR(mu,nbjet) cone; only if #mu >1

# Fully-hadronic MssmHbb
hltMssmHbbmonitoringAL100 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringAL100.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/pt100')
hltMssmHbbmonitoringAL100.FolderName = cms.string('HLT/HIG/MssmHbb/fullhadronic/pt100')
hltMssmHbbmonitoringAL100.nmuons = cms.uint32(0)
hltMssmHbbmonitoringAL100.nbjets = cms.uint32(2)
hltMssmHbbmonitoringAL100.bjetSelection = cms.string('pt>110 & abs(eta)<2.2')
hltMssmHbbmonitoringAL100.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringAL100.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

hltMssmHbbmonitoringAL116 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringAL116.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/pt116')
hltMssmHbbmonitoringAL116.FolderName = cms.string('HLT/HIG/MssmHbb/fullhadronic/pt116')
hltMssmHbbmonitoringAL116.nmuons = cms.uint32(0)
hltMssmHbbmonitoringAL116.nbjets = cms.uint32(2)
hltMssmHbbmonitoringAL116.bjetSelection = cms.string('pt>116 & abs(eta)<2.2')
hltMssmHbbmonitoringAL116.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringAL116.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

hltMssmHbbmonitoringAL128 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringAL128.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/pt128')
hltMssmHbbmonitoringAL128.FolderName = cms.string('HLT/HIG/MssmHbb/fullhadronic/pt128')
hltMssmHbbmonitoringAL128.nmuons = cms.uint32(0)
hltMssmHbbmonitoringAL128.nbjets = cms.uint32(2)
hltMssmHbbmonitoringAL128.bjetSelection = cms.string('pt>128 & abs(eta)<2.2')
hltMssmHbbmonitoringAL128.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringAL128.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

# Semi-leptonic MssmHbb(mu)
hltMssmHbbmonitoringSL40 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringSL40.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/pt40')
hltMssmHbbmonitoringSL40.FolderName = cms.string('HLT/HIG/MssmHbb/semileptonic/pt40')
hltMssmHbbmonitoringSL40.nmuons = cms.uint32(1)
hltMssmHbbmonitoringSL40.nbjets = cms.uint32(2)
hltMssmHbbmonitoringSL40.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
hltMssmHbbmonitoringSL40.bjetSelection = cms.string('pt>40 & abs(eta)<2.2')
hltMssmHbbmonitoringSL40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets40MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringSL40.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

hltMssmHbbmonitoringSL54 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringSL54.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/pt54')
hltMssmHbbmonitoringSL54.FolderName = cms.string('HLT/HIG/MssmHbb/semileptonic/pt54')
hltMssmHbbmonitoringSL54.nmuons = cms.uint32(1)
hltMssmHbbmonitoringSL54.nbjets = cms.uint32(2)
hltMssmHbbmonitoringSL54.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
hltMssmHbbmonitoringSL54.bjetSelection = cms.string('pt>54 & abs(eta)<2.2')
hltMssmHbbmonitoringSL54.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets54MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringSL54.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

hltMssmHbbmonitoringSL62 = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringSL62.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/pt62')
hltMssmHbbmonitoringSL62.FolderName = cms.string('HLT/HIG/MssmHbb/semileptonic/pt62')
hltMssmHbbmonitoringSL62.nmuons = cms.uint32(1)
hltMssmHbbmonitoringSL62.nbjets = cms.uint32(2)
hltMssmHbbmonitoringSL62.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
hltMssmHbbmonitoringSL62.bjetSelection = cms.string('pt>62 & abs(eta)<2.2')
hltMssmHbbmonitoringSL62.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets62MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v*')
hltMssmHbbmonitoringSL62.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

#control b-tagging 
hltMssmHbbmonitoringControl = hltMssmHbbmonitoring.clone()
#hltMssmHbbmonitoringControl.FolderName = cms.string('HLT/Higgs/MssmHbb/control/mu12_pt30_nobtag')
hltMssmHbbmonitoringControl.FolderName = cms.string('HLT/HIG/MssmHbb/control/mu12_pt30_nobtag')
hltMssmHbbmonitoringControl.nmuons = cms.uint32(1)
hltMssmHbbmonitoringControl.nbjets = cms.uint32(0)
hltMssmHbbmonitoringControl.njets = cms.uint32(1)
hltMssmHbbmonitoringControl.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrac\
k.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
hltMssmHbbmonitoringControl.jetSelection = cms.string('pt>40 & abs(eta)<2.2')
hltMssmHbbmonitoringControl.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_SingleJet30_Mu12_SinglePFJet40_v*')
hltMssmHbbmonitoringControl.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)


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
