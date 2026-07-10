import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

##############################
# Unconverted Hardware Value #
##############################

# subsets of hardware values which are converted to physical values
# by default, L1ScoutingNano will store physical values

# Muon
l1scoutingMuonUnconvertedVariables = cms.PSet(
    # use hwPt instead of hwEt in order to match the branch names used in the L1T-DPG NanoAOD flavour
    hwPt = Var("hwPt()", "int16", doc="hardware pt"),
    hwEta = Var("hwEta()", "int16", doc="hardware eta"),
    hwPhi = Var("hwPhi()", "int16", doc="hardware phi"),
    hwPtUnconstrained = Var("hwPtUnconstrained", "int16", doc="hardware unconstrained pt"),
    hwEtaAtVtx = Var("hwEtaAtVtx()", "int16", doc="hardware eta extrapolated at beam line"),
    hwPhiAtVtx = Var("hwPhiAtVtx()", "int16", doc="hardware phi extrapolated at beam line"),
)

# Calo objects
l1scoutingCaloObjectUnconvertedVariables = cms.PSet(
    # use hwPt instead of hwEt in order to match the branch names used in the L1T-DPG NanoAOD flavour
    hwPt = Var("hwEt()", "int16", doc="hardware pt"),
    hwEta = Var("hwEta()", "int16", doc="hardware eta"),
    hwPhi = Var("hwPhi()", "int16", doc="hardware phi"),
)

# CaloTowers
l1scoutingCaloTowerUnconvertedVariables = cms.PSet(
    hwEt = Var("hwEt()", "int16", doc="hardware Et"),
    hwEta = Var("hwEta()", "int16", doc="hardware eta"),
    hwPhi = Var("hwPhi()", "int16", doc="hardware phi"),
)

#################################################
# Physical Value Conversion from Hardware Value #
#################################################

# Conversions for muon
l1scoutingMuonConversions = cms.PSet(
    fPt = cms.PSet(func=cms.string("ugmt::fPt"), arg=cms.string("hwPt")),
    fEta = cms.PSet(func=cms.string("ugmt::fEta"), arg=cms.string("hwEta")),
    fPhi = cms.PSet(func=cms.string("ugmt::fPhi"), arg=cms.string("hwPhi")),
    fPtUnconstrained = cms.PSet(func=cms.string("ugmt::fPtUnconstrained"), arg=cms.string("hwPtUnconstrained")),
    fEtaAtVtx = cms.PSet(func=cms.string("ugmt::fEtaAtVtx"), arg=cms.string("hwEtaAtVtx")),
    fPhiAtVtx = cms.PSet(func=cms.string("ugmt::fPhiAtVtx"), arg=cms.string("hwPhiAtVtx"))
)

l1scoutingMuonPhysicalValueMap = cms.EDProducer("L1ScoutingMuonPhysicalValueMapProducer",
    src = cms.InputTag("l1ScGmtUnpacker", "Muon"),
    conversions = l1scoutingMuonConversions
)

# Conversions for Calo objects (EGamma, Tau, Jet)
l1scoutingCaloObjectConversions = cms.PSet(
    fEt = cms.PSet(func=cms.string("demux::fEt"), arg=cms.string("hwEt")),
    fEta = cms.PSet(func=cms.string("demux::fEta"), arg=cms.string("hwEta")),
    fPhi = cms.PSet(func=cms.string("demux::fPhi"), arg=cms.string("hwPhi")),
)

l1scoutingEGammaPhysicalValueMap = cms.EDProducer("L1ScoutingEGammaPhysicalValueMapProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "EGamma"),
    conversions = l1scoutingCaloObjectConversions
)

l1scoutingTauPhysicalValueMap = cms.EDProducer("L1ScoutingTauPhysicalValueMapProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "Tau"),
    conversions = l1scoutingCaloObjectConversions
)

l1scoutingJetPhysicalValueMap = cms.EDProducer("L1ScoutingJetPhysicalValueMapProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "Jet"),
    conversions = l1scoutingCaloObjectConversions
)

# Physical values (Et, Eta, Phi) for CaloTowers (Calo Layer-1)
l1scoutingCaloTowerPhysicalValueMap = cms.EDProducer("L1ScoutingCaloTowerPhysicalValueMapProducer",
    src = cms.InputTag("l1ScCaloTowerUnpacker", "CaloTower")
)

####################
# Table Definition #
####################

# default event content for L1ScoutingNano
# by default, L1ScoutingNano will store physical values

# Muon
l1scoutingMuonTable = cms.EDProducer("SimpleL1ScoutingMuonOrbitFlatTableProducer",
    src = cms.InputTag("l1ScGmtUnpacker", "Muon"),
    name = cms.string("L1Mu"),
    doc = cms.string("Muons from GMT"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        hwCharge = Var("hwCharge()", "int16", doc="charge (0 = invalid)"),
        hwQual = Var("hwQual()", "int16", doc="hardware quality"),
        tfMuonIndex = Var("tfMuonIndex()", "uint16",
            doc="index of muon at the uGMT input. 3 indices per link/sector/wedge. EMTF+ are 0-17, OMTF+ are 18-35, BMTF are 36-71, OMTF- are 72-89, EMTF- are 90-107"),
        hwDXY = Var("hwDXY()", "uint16", doc="hardware dxy"),
    ),
    externalVariables = cms.PSet(
        pt = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fPt"), "float", doc="pt"),
        eta = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fEta"), "float", doc="eta"),
        phi = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fPhi"), "float", doc="phi"),
        ptUnconstrained = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fPtUnconstrained"), "float", doc="unconstrained pt"),
        etaAtVtx = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fEtaAtVtx"), "float", doc="eta extrapolated at beam line"),
        phiAtVtx = ExtVar(cms.InputTag("l1scoutingMuonPhysicalValueMap", "fPhiAtVtx"), "float", doc="phi extrapolated at beam line"),
    ),
)

# EGamma
l1scoutingEGammaTable = cms.EDProducer("SimpleL1ScoutingEGammaOrbitFlatTableProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "EGamma"),
    name = cms.string("L1EG"),
    doc = cms.string("EGammas from Calo Demux"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        hwIso = Var("hwIso()", "int16", doc="hardware isolation (trigger units)")
    ),
    externalVariables = cms.PSet(
        pt = ExtVar(cms.InputTag("l1scoutingEGammaPhysicalValueMap", "fEt"), "float", doc="pt"),
        eta = ExtVar(cms.InputTag("l1scoutingEGammaPhysicalValueMap", "fEta"), "float", doc="eta"),
        phi = ExtVar(cms.InputTag("l1scoutingEGammaPhysicalValueMap", "fPhi"), "float", doc="phi"),
    ),
)

# Tau
l1scoutingTauTable = cms.EDProducer("SimpleL1ScoutingTauOrbitFlatTableProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "Tau"),
    name = cms.string("L1Tau"),
    doc = cms.string("Taus from Calo Demux"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        hwIso = Var("hwIso()", "int16", doc="hardware isolation (trigger units)")
    ),
    externalVariables = cms.PSet(
        pt = ExtVar(cms.InputTag("l1scoutingTauPhysicalValueMap", "fEt"), "float", doc="pt"),
        eta = ExtVar(cms.InputTag("l1scoutingTauPhysicalValueMap", "fEta"), "float", doc="eta"),
        phi = ExtVar(cms.InputTag("l1scoutingTauPhysicalValueMap", "fPhi"), "float", doc="phi"),
    ),
)

# Jet
l1scoutingJetTable = cms.EDProducer("SimpleL1ScoutingJetOrbitFlatTableProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "Jet"),
    name = cms.string("L1Jet"),
    doc = cms.string("Jets from Calo Demux"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    skipNonExistingSrc = cms.bool(False),
    variables = cms.PSet(
        hwQual = Var("hwQual()", "int16", doc="hardware quality"),
    ),
    externalVariables = cms.PSet(
        pt = ExtVar(cms.InputTag("l1scoutingJetPhysicalValueMap", "fEt"), "float", doc="pt"),
        eta = ExtVar(cms.InputTag("l1scoutingJetPhysicalValueMap", "fEta"), "float", doc="eta"),
        phi = ExtVar(cms.InputTag("l1scoutingJetPhysicalValueMap", "fPhi"), "float", doc="phi"),
    ),
)

# EtSum
l1scoutingEtSumTable = cms.EDProducer("L1ScoutingEtSumOrbitFlatTableProducer",
    src = cms.InputTag("l1ScCaloUnpacker", "EtSum"),
    name = cms.string("L1EtSum"),
    doc = cms.string("EtSums from Calo Demux"),
    singleton = cms.bool(False),
    writePhysicalValues = cms.bool(True),
    writeHardwareValues = cms.bool(False),
    writeHF = cms.bool(True),
    writeAsym = cms.bool(False),
    writeMinBias = cms.bool(False),
    writeTowerCount = cms.bool(True),
    writeCentrality = cms.bool(False),
)

# BMTF Stub
l1scoutingBMTFStubTable = cms.EDProducer("SimpleL1ScoutingBMTFStubOrbitFlatTableProducer",
    src = cms.InputTag("l1ScBMTFUnpacker", "BMTFStub"),
    name = cms.string("L1BMTFStub"),
    doc = cms.string("Stubs from BMTF"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        hwPhi = Var("hwPhi()", "int16", doc="hardware phi (raw L1T units)"),
        hwPhiB = Var("hwPhiB()", "int16", doc="hardware phiB (raw L1T units)"),
        hwQual = Var("hwQual()", "int16", doc="hardware quality (raw L1T units)"),
        hwEta = Var("hwEta()", "int16", doc="hardware eta (raw L1T units)"),
        hwQEta = Var("hwQEta()", "int16", doc="hardware Qeta (raw L1T units)"),
        station = Var("station()", "int16", doc="station (raw L1T units)"),
        wheel = Var("wheel()", "int16", doc="wheel (raw L1T units)"),
        sector = Var("sector()", "int16", doc="sector (raw L1T units)"),
        tag = Var("tag()", "int16", doc="tag (raw L1T units)"),
    ),
)

# CaloTowers
l1scoutingCaloTowerTable = cms.EDProducer("SimpleL1ScoutingCaloTowerOrbitFlatTableProducer",
    src = cms.InputTag("l1ScCaloTowerUnpacker", "CaloTower"),
    name = cms.string("L1CaloTower"),
    doc = cms.string("CaloTowers from Calo Layer-1"),
    singleton = cms.bool(False),
    skipNonExistingSrc = cms.bool(False),
    variables = cms.PSet(
        erBits = Var("erBits()", "int16", doc="hardware energy-ratio bits"),
        miscBits = Var("miscBits()", "int16", doc="hardware misc-bits"),
    ),
    externalVariables = cms.PSet(
        pt = ExtVar(cms.InputTag("l1scoutingCaloTowerPhysicalValueMap", "fEt"), "float", doc="pt", precision=10),
        eta = ExtVar(cms.InputTag("l1scoutingCaloTowerPhysicalValueMap", "fEta"), "float", doc="eta", precision=10),
        phi = ExtVar(cms.InputTag("l1scoutingCaloTowerPhysicalValueMap", "fPhi"), "float", doc="phi", precision=10),
    )
)
