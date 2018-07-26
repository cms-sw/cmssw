import FWCore.ParameterSet.Config as cms

L1TkCaloHTMiss = cms.EDProducer("L1TkHTMissProducer",
     L1TkJetInputTag = cms.InputTag("L1TkCaloJets", "L1TkCaloJets"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     jet_maxEta = cms.double(2.2),          # maximum eta of jets for HT
     jet_minPt = cms.double(15.0),          # minimum pt of jets for HT [GeV]
     DoVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint
     DeltaZ = cms.double(1.0),              # require jets to have |z_jet - z_ref| below DeltaZ [cm]
     PrimaryVtxConstrain = cms.bool(False), # use primary vertex instead of leading jet as reference z position
     UseCaloJets = cms.bool(True)        # determines whether matched jets or standalone jets are used for MHT
)

L1TkCaloHTMissVtx = L1TkCaloHTMiss.clone()
L1TkCaloHTMiss.DoVtxConstrain = cms.bool(True)

L1TrackerHTMiss = L1TkCaloHTMiss.clone()
L1TrackerHTMiss.L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets")
L1TrackerHTMiss.jet_maxEta = cms.double(2.4)
L1TrackerHTMiss.jet_minPt = cms.double(15.0)
L1TrackerHTMiss.UseCaloJets = cms.bool(False)



