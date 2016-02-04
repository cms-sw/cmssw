import FWCore.ParameterSet.Config as cms

#Muons
# Clone L1 muons
CloneL1ExtraMuons = cms.EDProducer("L1MuonParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles")
)

# Select L1 muons
SelectL1Muons = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraMuons"),
    ptMin = cms.double(20.0)
)


# Iso EM
# Clone L1 isolated EM
CloneL1ExtraIsoEm = cms.EDProducer("L1EmParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles","Isolated")
)

# Select L1 isolated EM
SelectL1IsoEm = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraIsoEm"),
    ptMin = cms.double(10.0)
)

# Non-iso EM
# Clone L1 isolated EM
CloneL1ExtraNonIsoEm = cms.EDProducer("L1EmParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles","NonIsolated")
)

# Select L1 isolated EM
SelectL1NonIsoEm = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraNonIsoEm"),
    ptMin = cms.double(10.0)
)

#Merge L1 isolated EM and L1 non-isolated EM
MergeL1Em = cms.EDProducer("CandViewMerger",
    src = cms.VInputTag("CloneL1ExtraIsoEm","CloneL1ExtraNonIsoEm")
)

# Select L1 EM
SelectL1Em = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("MergeL1Em"),
    ptMin = cms.double(10.0)
) 


# Cen Jets
# Clone L1 central jets
CloneL1ExtraCenJets = cms.EDProducer("L1JetParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles","Central")
)

# Select L1 central jets
SelectL1CenJets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraCenJets"),
    ptMin = cms.double(10.0)
)

# For Jets
# Clone L1 forward jets
CloneL1ExtraForJets = cms.EDProducer("L1JetParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles","Forward")
)

# Select L1 forward jets
SelectL1ForJets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraForJets"),
    ptMin = cms.double(10.0)
)

# Tau Jets
# Clone L1 tau jets
CloneL1ExtraTauJets = cms.EDProducer("L1JetParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles","Tau")
)

# Select L1 tau jets
SelectL1TauJets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraTauJets"),
    ptMin = cms.double(10.0)
)

#Merge L1 tau jets and L1 central jets
MergeL1CenTauJets = cms.EDProducer("CandViewMerger",
    src = cms.VInputTag("CloneL1ExtraTauJets","CloneL1ExtraCenJets")
)

SelectL1CenTauJets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("MergeL1CenTauJets"),
    ptMin = cms.double(10.0)
)                                       

#Merge L1 tau jets, central jets, and forward jets
MergeL1Jets = cms.EDProducer("CandViewMerger",
    src = cms.VInputTag("CloneL1ExtraTauJets","CloneL1ExtraCenJets","CloneL1ExtraForJets")
)

# Select L1 jets
SelectL1Jets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("MergeL1Jets"),
    ptMin = cms.double(10.0)
)

# Missing Et
# Clone L1 Met
CloneL1ExtraMet = cms.EDProducer("L1EtMissParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles", "MET")
)

# Select L1 MET
SelectL1Met = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraMet"),
    ptMin = cms.double(10.0)
)

# Clone L1 MHt
CloneL1ExtraMHt = cms.EDProducer("L1EtMissParticleShallowCloneProducer",
    src = cms.InputTag("hltL1extraParticles", "MHT")
)

# Select L1 MHt
SelectL1MHt = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneL1ExtraMHt"),
    ptMin = cms.double(10.0)
)


L1MuonSelection = cms.Sequence(CloneL1ExtraMuons*SelectL1Muons)

L1IsoEmSelection = cms.Sequence(CloneL1ExtraIsoEm*SelectL1IsoEm)
L1NonIsoEmSelection = cms.Sequence(CloneL1ExtraNonIsoEm*SelectL1NonIsoEm)
L1EmSelection = cms.Sequence(CloneL1ExtraIsoEm*CloneL1ExtraNonIsoEm*MergeL1Em*SelectL1Em)

L1CenJetSelection = cms.Sequence(CloneL1ExtraCenJets*SelectL1CenJets)
L1ForJetSelection = cms.Sequence(CloneL1ExtraForJets*SelectL1ForJets)
L1TauJetSelection = cms.Sequence(CloneL1ExtraTauJets*SelectL1TauJets)
L1CenTauJetSelection = cms.Sequence(CloneL1ExtraTauJets*CloneL1ExtraCenJets*MergeL1CenTauJets*SelectL1CenTauJets)
L1JetSelection = cms.Sequence(CloneL1ExtraTauJets*CloneL1ExtraCenJets*CloneL1ExtraForJets*MergeL1Jets*SelectL1Jets)

L1MetSelection = cms.Sequence(CloneL1ExtraMet*SelectL1Met)
L1MHtSelection = cms.Sequence(CloneL1ExtraMHt*SelectL1MHt)


