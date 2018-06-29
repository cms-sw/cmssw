import FWCore.ParameterSet.Config as cms

L1TkMatchedHTMiss = cms.EDProducer("L1TkHTMissProducer",
     L1TkJetInputTag = cms.InputTag("L1TkMatchedJets", "L1TkMatchedJets"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     jet_maxEta = cms.double(2.2),          # maximum eta of jets for HT
     jet_minPt = cms.double(15.0),          # minimum pt of jets for HT [GeV]
     DoVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint
     DeltaZ = cms.double(1.0),              # require jets to have |z_jet - z_ref| below DeltaZ [cm]
     PrimaryVtxConstrain = cms.bool(False), # use primary vertex instead of leading jet as reference z position
     UseMatchedJets = cms.bool(True)        # determines whether matched jets or standalone jets are used for MHT
)

L1TkMatchedHTMissVtx = L1TkMatchedHTMiss.clone()
L1TkMatchedHTMiss.DoVtxConstrain = cms.bool(True)

L1TkHTMiss = L1TkMatchedHTMiss.clone()
L1TkHTMiss.L1TkJetInputTag = cms.InputTag("L1TkJets","L1TkJets")
L1TkHTMiss.UseMatchedJets = cms.bool(False)
