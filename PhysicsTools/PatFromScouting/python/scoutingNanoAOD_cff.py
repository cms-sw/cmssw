"""
Scouting NanoAOD configuration.

Produces NanoAOD flat tables directly from scouting MiniAOD collections,
bypassing the standard NanoAOD jet update chains that require full reconstruction.

Tables produced:
- Muon: pt, eta, phi, mass, charge, isolation, ID flags
- MuonNoVtx: displaced muons without vertex constraint (2024+)
- Electron: pt, eta, phi, mass, charge, shower shape, isolation
- Photon: pt, eta, phi, mass, shower shape, isolation
- Jet: pt, eta, phi, mass, energy fractions, multiplicities, b-tags
- MET: pt, phi, sumEt
- PV: x, y, z, errors, chi2, ndof
- DimuonVtx: displaced dimuon vertices with muon index cross-references
- DimuonVtxNoVtx: displaced dimuon vertices from NoVtx muons (2024+)
- Event: fixedGridRhoAll

To include HLT trigger decisions, add to outputCommands:
    'keep edmTriggerResults_*_*_*'
This will create HLT_* and DST_* boolean branches for all trigger paths.
"""

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run3_scouting_2024_cff import run3_scouting_2024
from PhysicsTools.NanoAOD.common_cff import Var

# Muon table
scoutingMuonTable = cms.EDProducer("SimplePATMuonFlatTableProducer",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(""),
    name = cms.string("Muon"),
    doc = cms.string("Muons from scouting"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="mass", precision=10),
        charge = Var("charge", int, doc="charge"),
        pdgId = Var("pdgId", int, doc="PDG ID"),
        # Use global track dxy/dz - for scouting muons the track is embedded as combinedMuon
        dxy = Var("? globalTrack.isNonnull ? globalTrack.dxy : 0", float, doc="dxy w.r.t. track reference point", precision=10),
        dxyErr = Var("? globalTrack.isNonnull ? globalTrack.dxyError : 0", float, doc="dxy uncertainty", precision=10),
        dz = Var("? globalTrack.isNonnull ? globalTrack.dz : 0", float, doc="dz w.r.t. track reference point", precision=10),
        dzErr = Var("? globalTrack.isNonnull ? globalTrack.dzError : 0", float, doc="dz uncertainty", precision=10),
        trkChi2 = Var("? globalTrack.isNonnull ? globalTrack.normalizedChi2 : -1", float, doc="track chi2/ndof", precision=6),
        nValidHits = Var("? globalTrack.isNonnull ? globalTrack.numberOfValidHits : -1", int, doc="number of valid hits"),
        # ID flags
        isGlobal = Var("isGlobalMuon", bool, doc="is global muon"),
        isTracker = Var("isTrackerMuon", bool, doc="is tracker muon"),
        isStandalone = Var("isStandAloneMuon", bool, doc="is standalone muon"),
        isPF = Var("isPFMuon", bool, doc="is PF muon"),
        # Station and layer counts
        nStations = Var("numberOfMatchedStations()", int, doc="number of matched stations with default arbitration"),
        nTrackerLayers = Var("numberOfTrackerLayersWithMeasurement()", int, doc="number of tracker layers with measurement"),
        nPixelLayers = Var("numberOfPixelLayersWithMeasurement()", int, doc="number of pixel layers with measurement"),
        nChambers = Var("numberOfChambers()", int, doc="number of chambers"),
        nChambersCSCorDT = Var("numberOfChambersCSCorDT()", int, doc="number of CSC or DT chambers"),
        # Hit counts
        nValidMuonHits = Var("numberOfValidMuonHits()", int, doc="number of valid muon hits"),
        nValidPixelHits = Var("numberOfValidPixelHits()", int, doc="number of valid pixel hits"),
        nValidStripHits = Var("numberOfValidStripHits()", int, doc="number of valid strip hits"),
        # Isolation - scouting provides trackIso, ecalIso, hcalIso (not PF components)
        # Use calorimeter isolation sum as proxy for relIso
        relIso = Var("(isolationR03().emEt + isolationR03().hadEt)/pt", float, doc="relative calo isolation (ecal+hcal)/pt", precision=6),
        ecalIso = Var("isolationR03().emEt", float, doc="ECAL isolation", precision=6),
        hcalIso = Var("isolationR03().hadEt", float, doc="HCAL isolation", precision=6),
        trkIso = Var("isolationR03().sumPt", float, doc="tracker isolation", precision=6),
        tkRelIso = Var("isolationR03().sumPt/pt", float, doc="tracker relative isolation", precision=6),
    )
)

# NoVtx muon table (2024+ only)
scoutingMuonNoVtxTable = scoutingMuonTable.clone(
    src = cms.InputTag("slimmedMuonsNoVtx"),
    name = cms.string("MuonNoVtx"),
    doc = cms.string("Displaced muons from scouting (no vertex constraint)"),
)

# Electron table
scoutingElectronTable = cms.EDProducer("SimplePATElectronFlatTableProducer",
    src = cms.InputTag("slimmedElectrons"),
    cut = cms.string(""),
    name = cms.string("Electron"),
    doc = cms.string("Electrons from scouting"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="mass", precision=10),
        charge = Var("charge", int, doc="charge"),
        pdgId = Var("pdgId", int, doc="PDG ID"),
        dxy = Var("dB('PV2D')", float, doc="dxy wrt PV", precision=10),
        dz = Var("dB('PVDZ')", float, doc="dz wrt PV", precision=10),
        # Shower shape
        sieie = Var("full5x5_sigmaIetaIeta", float, doc="sigma_IetaIeta", precision=10),
        hoe = Var("hadronicOverEm", float, doc="H/E", precision=8),
        # ID
        dEtaIn = Var("deltaEtaSuperClusterTrackAtVtx", float, doc="dEta(SC, track)", precision=10),
        dPhiIn = Var("deltaPhiSuperClusterTrackAtVtx", float, doc="dPhi(SC, track)", precision=10),
        # Isolation
        pfRelIso03_all = Var("(pfIsolationVariables().sumChargedHadronPt + max(pfIsolationVariables().sumNeutralHadronEt + pfIsolationVariables().sumPhotonEt - 0.5*pfIsolationVariables().sumPUPt, 0.0))/pt", float, doc="PF relative isolation dR=0.3", precision=6),
        ecalIso = Var("ecalPFClusterIso", float, doc="ECAL PF cluster isolation", precision=6),
        hcalIso = Var("hcalPFClusterIso", float, doc="HCAL PF cluster isolation", precision=6),
        trkIso = Var("trackIso", float, doc="tracker isolation", precision=6),
    )
)

# Photon table
scoutingPhotonTable = cms.EDProducer("SimplePATPhotonFlatTableProducer",
    src = cms.InputTag("slimmedPhotons"),
    cut = cms.string(""),
    name = cms.string("Photon"),
    doc = cms.string("Photons from scouting"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="mass", precision=10),
        # Shower shape - use userFloats from our producer
        sieie = Var("userFloat('sigmaIetaIeta')", float, doc="sigma_IetaIeta", precision=10),
        hoe = Var("userFloat('hOverE')", float, doc="H/E", precision=8),
        # Isolation from userFloats
        ecalIso = Var("userFloat('ecalIso')", float, doc="ECAL isolation", precision=6),
        hcalIso = Var("userFloat('hcalIso')", float, doc="HCAL isolation", precision=6),
    )
)

# Jet table
scoutingJetTable = cms.EDProducer("SimplePATJetFlatTableProducer",
    src = cms.InputTag("slimmedJets"),
    cut = cms.string(""),
    name = cms.string("Jet"),
    doc = cms.string("Jets from scouting (AK4 PF jets)"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="mass", precision=10),
        area = Var("jetArea", float, doc="jet area", precision=6),
        # Energy fractions
        chHEF = Var("chargedHadronEnergyFraction", float, doc="charged hadron energy fraction", precision=6),
        neHEF = Var("neutralHadronEnergyFraction", float, doc="neutral hadron energy fraction", precision=6),
        chEmEF = Var("chargedEmEnergyFraction", float, doc="charged EM energy fraction", precision=6),
        neEmEF = Var("neutralEmEnergyFraction", float, doc="neutral EM energy fraction", precision=6),
        muEF = Var("muonEnergyFraction", float, doc="muon energy fraction", precision=6),
        # Multiplicities
        nConstituents = Var("numberOfDaughters", int, doc="number of constituents"),
        chMultiplicity = Var("chargedMultiplicity", int, doc="charged multiplicity"),
        neMultiplicity = Var("neutralMultiplicity", int, doc="neutral multiplicity"),
        # B-tagging (from scouting)
        btagCSV = Var("bDiscriminator('pfCombinedSecondaryVertexV2BJetTags')", float, doc="CSV b-tag discriminator", precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')", float, doc="DeepCSV b-tag discriminator", precision=10),
    )
)

# MET table
scoutingMETTable = cms.EDProducer("SimplePATMETFlatTableProducer",
    src = cms.InputTag("slimmedMETs"),
    name = cms.string("MET"),
    doc = cms.string("MET from scouting PF candidates"),
    singleton = cms.bool(True),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="MET pt", precision=10),
        phi = Var("phi", float, doc="MET phi", precision=10),
        sumEt = Var("sumEt", float, doc="scalar sum of Et", precision=10),
    )
)

# Primary vertex table
scoutingPVTable = cms.EDProducer("SimpleVertexFlatTableProducer",
    src = cms.InputTag("offlineSlimmedPrimaryVertices"),
    cut = cms.string(""),
    name = cms.string("PV"),
    doc = cms.string("Primary vertices from scouting"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        x = Var("x", float, doc="x position", precision=10),
        y = Var("y", float, doc="y position", precision=10),
        z = Var("z", float, doc="z position", precision=10),
        ndof = Var("ndof", float, doc="number of degrees of freedom", precision=6),
        chi2 = Var("chi2", float, doc="chi2", precision=6),
        # Error
        xErr = Var("xError", float, doc="x position error", precision=10),
        yErr = Var("yError", float, doc="y position error", precision=10),
        zErr = Var("zError", float, doc="z position error", precision=10),
    )
)

# L1 trigger bits conversion (GlobalAlgBlk -> TriggerResults)
# This converts the gtStage2Digis output to a format NanoAOD can use
l1bits = cms.EDProducer("L1TriggerResultsConverter",
    src = cms.InputTag("gtStage2Digis"),
    legacyL1 = cms.bool(False),
    storeUnprefireableBits = cms.bool(True),
)

# Event-level variables
scoutingEventTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string(""),
    variables = cms.PSet(
        fixedGridRhoFastjetAll = cms.PSet(
            src = cms.InputTag("fixedGridRhoFastjetAll"),
            doc = cms.string("rho from HLT scouting"),
            type = cms.string("double"),
            precision = cms.int32(8)
        ),
    )
)

# Dimuon displaced vertex table
scoutingDimuonVtxTable = cms.EDProducer("SimpleSecondaryVertexFlatTableProducer",
    src = cms.InputTag("scoutingDimuonVertices"),
    cut = cms.string(""),
    name = cms.string("DimuonVtx"),
    doc = cms.string("Displaced dimuon vertices from scouting"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="dimuon invariant mass", precision=10),
        x = Var("position().x()", float, doc="vertex X position, in cm", precision=10),
        y = Var("position().y()", float, doc="vertex Y position, in cm", precision=10),
        z = Var("position().z()", float, doc="vertex Z position, in cm", precision=14),
        ndof = Var("vertexNdof()", float, doc="number of degrees of freedom", precision=8),
        chi2 = Var("vertexNormalizedChi2()", float, doc="reduced chi2, i.e. chi/ndof", precision=8),
        nMuons = Var("numberOfDaughters()", "uint8", doc="number of daughter muons"),
        mu1Idx = Var("?numberOfDaughters()>0?daughterPtr(0).key():-1", "int16", doc="index of first muon in Muon collection"),
        mu2Idx = Var("?numberOfDaughters()>1?daughterPtr(1).key():-1", "int16", doc="index of second muon in Muon collection"),
    ),
)

# Dimuon displaced vertex table for NoVtx muons (2024+ only)
scoutingDimuonVtxNoVtxTable = cms.EDProducer("SimpleSecondaryVertexFlatTableProducer",
    src = cms.InputTag("scoutingDimuonVerticesNoVtx"),
    cut = cms.string(""),
    name = cms.string("DimuonVtxNoVtx"),
    doc = cms.string("Displaced dimuon vertices from scouting (NoVtx muons)"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=12),
        phi = Var("phi", float, doc="phi", precision=12),
        mass = Var("mass", float, doc="dimuon invariant mass", precision=10),
        x = Var("position().x()", float, doc="vertex X position, in cm", precision=10),
        y = Var("position().y()", float, doc="vertex Y position, in cm", precision=10),
        z = Var("position().z()", float, doc="vertex Z position, in cm", precision=14),
        ndof = Var("vertexNdof()", float, doc="number of degrees of freedom", precision=8),
        chi2 = Var("vertexNormalizedChi2()", float, doc="reduced chi2, i.e. chi/ndof", precision=8),
        nMuons = Var("numberOfDaughters()", "uint8", doc="number of daughter muons"),
        mu1Idx = Var("?numberOfDaughters()>0?daughterPtr(0).key():-1", "int16", doc="index of first muon in MuonNoVtx collection"),
        mu2Idx = Var("?numberOfDaughters()>1?daughterPtr(1).key():-1", "int16", doc="index of second muon in MuonNoVtx collection"),
    ),
)

# Scouting NanoAOD task - core tables
scoutingNanoAODTask = cms.Task(
    scoutingMuonTable,
    scoutingElectronTable,
    scoutingPhotonTable,
    scoutingJetTable,
    scoutingMETTable,
    scoutingPVTable,
    scoutingEventTable,
    scoutingDimuonVtxTable,
    l1bits,
)

# For 2024+, add NoVtx muon table and its dimuon vertex table
_scoutingNanoAODTask_2024 = scoutingNanoAODTask.copy()
_scoutingNanoAODTask_2024.add(scoutingMuonNoVtxTable)
_scoutingNanoAODTask_2024.add(scoutingDimuonVtxNoVtxTable)
run3_scouting_2024.toReplaceWith(scoutingNanoAODTask, _scoutingNanoAODTask_2024)

scoutingNanoAODSequence = cms.Sequence(scoutingNanoAODTask)


# ============================================================
# Customization function for cmsDriver
# ============================================================

def customiseScoutingNanoAOD(process):
    """
    Customization function to add scouting NanoAOD production.

    Usage with cmsDriver (two-step workflow):

    Step 1 - Scouting RAW to MiniAOD:
        cmsRun test_scoutingToMiniAOD_cfg.py

    Step 2 - MiniAOD to NanoAOD:
        cmsDriver.py NANO --conditions auto:run3_data_prompt \\
            --era Run3 --step NANO --datatier NANOAOD \\
            --filein file:scoutingToMiniAOD_test.root \\
            --fileout file:scoutingNanoAOD.root \\
            --customise PhysicsTools/PatFromScouting/scoutingNanoAOD_cff.customiseScoutingNanoAOD \\
            -n 100

    Or for standalone usage:
        from PhysicsTools.PatFromScouting.scoutingNanoAOD_cff import customiseScoutingNanoAOD
        process = customiseScoutingNanoAOD(process)
    """

    # When called via cmsDriver @ScoutMini, the task is already loaded
    # and scheduled. Only add modules if they are not yet in the process.
    if not hasattr(process, 'scoutingNanoAODTask'):
        process.load('PhysicsTools.PatFromScouting.scoutingNanoAOD_cff')

        process.scoutingNanoAOD_step = cms.Path()
        process.scoutingNanoAOD_step.associate(process.scoutingNanoAODTask)

        if hasattr(process, 'schedule') and process.schedule is not None:
            process.schedule.insert(0, process.scoutingNanoAOD_step)

    # Configure output
    if hasattr(process, 'NANOAODoutput'):
        process.NANOAODoutput.outputCommands = cms.untracked.vstring(
            'drop *',
            'keep nanoaodFlatTable_*_*_*',
            'keep edmTriggerResults_*_*_*',
            'keep nanoaodMergeableCounterTable_*_*_*',
            'keep nanoaodUniqueString_*_*_*',
        )

    return process
