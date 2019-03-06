import FWCore.ParameterSet.Config as cms

L1TkCaloHTMiss = cms.EDProducer("L1TkHTMissProducer",
     L1TkJetInputTag = cms.InputTag("L1TkCaloJets", "L1TkCaloJets"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     jet_maxEta = cms.double(2.2),          # maximum eta of jets for HT
     jet_minPt = cms.double(15.0),          # minimum pt of jets for HT [GeV]
     jet_minNtracksHighPt=cms.int32(0),       #Add track jet quality criteria pT>100
     jet_minNtracksLowPt=cms.int32(0),        #Add track jet quality criteria pT>50
     DoVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint32
     DeltaZ = cms.double(1.0),              # require jets to have |z_jet - z_ref| below DeltaZ [cm]
     PrimaryVtxConstrain = cms.bool(False), # use primary vertex instead of leading jet as reference z position
     UseCaloJets = cms.bool(True)        # determines whether matched jets or standalone jets are used for MHT

)

L1TrackerHTMiss = cms.EDProducer("L1TkHTMissProducer",
	L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets"),
	L1VertexInputTag = cms.InputTag("VertexProducer","l1vertextdr"),
	jet_maxEta = cms.double(2.4),
	jet_minPt = cms.double(5.0),
	jet_minNtracksLowPt=cms.int32(2),
	jet_minNtracksHighPt=cms.int32(3), 
        UseCaloJets = cms.bool(False),
        DoVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint32
        DeltaZ = cms.double(1.0),              # This is a dummy value for track only jets
        PrimaryVtxConstrain = cms.bool(False) # primary vertex already applied to track jet collections
)
