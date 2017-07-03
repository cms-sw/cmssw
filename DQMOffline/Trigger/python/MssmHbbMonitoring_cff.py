import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitoring_cfi import mssmHbbMonitoring

#Define MssmHbb specific cuts 
hltMssmHbbmonitoring =  mssmHbbMonitoring.clone()
hltMssmHbbmonitoring.btagalgo  = cms.InputTag("pfCombinedSecondaryVertexV2BJetTags")
hltMssmHbbmonitoring.workingpoint     = cms.double(0.92) # tight WP
hltMssmHbbmonitoring.bJetDeltaEtaMax = cms.double(1.6)   # deta cut between leading bjets
hltMssmHbbmonitoring.bJetMuDeltaRmax = cms.double(0.4)   # dR(mu,nbjet) cone; only if #mu >1

# Fully-hadronic MssmHbb
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.nmuons = cms.uint32(0)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.bjetSelection = cms.string('pt>110 & abs(eta)<2.2')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6_v*')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.nmuons = cms.uint32(0)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.bjetSelection = cms.string('pt>116 & abs(eta)<2.2')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6_v*')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/fullhadronic/DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.nmuons = cms.uint32(0)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.bjetSelection = cms.string('pt>128 & abs(eta)<2.2')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6_v*')
DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

# Semi-leptonic MssmHbb(mu)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.nmuons = cms.uint32(1)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.bjetSelection = cms.string('pt>40 & abs(eta)<2.2')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6_v*')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.nmuons = cms.uint32(1)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.bjetSelection = cms.string('pt>54 & abs(eta)<2.2')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6_v*')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6 = hltMssmHbbmonitoring.clone()
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.FolderName = cms.string('HLT/Higgs/MssmHbb/semileptonic/DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.nmuons = cms.uint32(1)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.nbjets = cms.uint32(2)
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.bjetSelection = cms.string('pt>62 & abs(eta)<2.2')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6_v*')
DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)

#control b-tagging 
SingleJet30_Mu12_SinglePFJet40 = hltMssmHbbmonitoring.clone()
SingleJet30_Mu12_SinglePFJet40.FolderName = cms.string('HLT/Higgs/MssmHbb/control/SingleJet30_Mu12_SinglePFJet40')
SingleJet30_Mu12_SinglePFJet40.nmuons = cms.uint32(1)
SingleJet30_Mu12_SinglePFJet40.nbjets = cms.uint32(0)
SingleJet30_Mu12_SinglePFJet40.njets = cms.uint32(1)
SingleJet30_Mu12_SinglePFJet40.muoSelection = cms.string('pt>12 & abs(eta)<2.2 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrac\
k.hitPattern.numberOfValidPixelHits>0 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10')
SingleJet30_Mu12_SinglePFJet40.jetSelection = cms.string('pt>40 & abs(eta)<2.2')
SingleJet30_Mu12_SinglePFJet40.numGenericTriggerEventPSet.hltPaths = cms.vstring('SingleJet30_Mu12_SinglePFJet40_v*')
SingleJet30_Mu12_SinglePFJet40.histoPSet.jetPtBinning = cms.vdouble(0,250,280,300,320,360,400,700,1000,1500)


mssmHbbMonitorHLT = cms.Sequence(
    #full-hadronic
    DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets110MaxDeta1p6
    + DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets116MaxDeta1p6
    + DoubleJets100_DoubleBtagCSV_0p92_DoublePFJets128MaxDeta1p6
    #semileptonic
    + DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets40MaxDeta1p6
    + DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets54MaxDeta1p6
    + DoubleJets30_Mu12_DoubleBtagCSV_0p92_DoublePFJets62MaxDeta1p6    

    #control no b-tag
    + SingleJet30_Mu12_SinglePFJet40
)
