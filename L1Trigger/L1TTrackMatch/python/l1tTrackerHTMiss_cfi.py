import FWCore.ParameterSet.Config as cms

l1tTkCaloHTMiss = cms.EDProducer("L1TkHTMissProducer",
     L1TkJetInputTag = cms.InputTag("l1tTkCaloJets", "L1TkCaloJets"),
     L1VertexInputTag = cms.InputTag("l1tVertexFinder", "L1Vertices"),
     jet_maxEta = cms.double(2.2),          # maximum eta of jets for HT
     jet_minPt = cms.double(15.0),          # minimum pt of jets for HT [GeV]
     jet_minNtracksHighPt=cms.int32(0),     #Add track jet quality criteria pT>100
     jet_minNtracksLowPt=cms.int32(0),      #Add track jet quality criteria pT>50
     jet_minJetEtLowPt=cms.double(0.0),     # Track jet quality criteria
     jet_minJetEtHighPt=cms.double(0.0),    
     doVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint32
     deltaZ = cms.double(1.0),              # require jets to have |z_jet - z_ref| below deltaZ [cm]
     primaryVtxConstrain = cms.bool(False), # use primary vertex instead of leading jet as reference z position
     useCaloJets = cms.bool(True),        # determines whether matched jets or standalone jets are used for MHT
     displaced = cms.bool(False) #Run with prompt/displaced jets - only useful for track jets
)

l1tTkCaloHTMissVtx = l1tTkCaloHTMiss.clone(doVtxConstrain = True)

l1tTrackerHTMiss = cms.EDProducer("L1TkHTMissProducer",
    L1TkJetInputTag = cms.InputTag("l1tTrackJets", "L1TrackJets"),
    L1VertexInputTag = cms.InputTag("l1tVertexFinder", "L1Vertices"),
    jet_maxEta = cms.double(2.4),
    jet_minPt = cms.double(5.0),
    jet_minNtracksLowPt=cms.int32(2),
    jet_minNtracksHighPt=cms.int32(3),
    jet_minJetEtLowPt=cms.double(50.0),     # Track jet quality criteria
    jet_minJetEtHighPt=cms.double(100.0),
    useCaloJets = cms.bool(False),
    doVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint32
    deltaZ = cms.double(1.0),              # This is a dummy value for track only jets
    primaryVtxConstrain = cms.bool(False), # primary vertex already applied to track jet collections
    displaced = cms.bool(False) # Run with prompt/displaced jets
)

l1tTrackerHTMissExtended = cms.EDProducer("L1TkHTMissProducer",
    L1TkJetInputTag = cms.InputTag("l1tTrackJetsExtended", "L1TrackJetsExtended"),
    L1VertexInputTag = cms.InputTag("l1tVertexFinder", "L1Vertices"),
    jet_maxEta = cms.double(2.4),
    jet_minPt = cms.double(5.0),
    jet_minNtracksLowPt=cms.int32(2),
    jet_minNtracksHighPt=cms.int32(3),
    jet_minJetEtLowPt=cms.double(50.0),     # Track jet quality criteria
    jet_minJetEtHighPt=cms.double(100.0),
    useCaloJets = cms.bool(False),
    doVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint32
    deltaZ = cms.double(1.0),              # This is a dummy value for track only jets
    primaryVtxConstrain = cms.bool(False), # primary vertex already applied to track jet collections
    displaced = cms.bool(True) # Run with prompt/displaced jets
)
