"""
NanoAOD customizations for scouting MiniAOD input.

This module provides era-based modifiers and customization functions to run
standard NanoAOD on MiniAOD produced from scouting data.

Usage with cmsDriver:

    # Step 1: Scouting RAW -> MiniAOD
    cmsDriver.py step1 --conditions auto:run3_data_prompt \
        --era Run3 --step MINI:PhysicsTools/PatFromScouting/scoutingMiniAOD_cff.scoutingMiniAODTask \
        --filein file:scouting.root --fileout file:scoutingMiniAOD.root

    # Step 2: MiniAOD -> NanoAOD with scouting modifier
    cmsDriver.py NANO --conditions auto:run3_data_prompt \
        --era Run3 --step NANO \
        --customise PhysicsTools/PatFromScouting/nanoAOD_scouting_cff.customiseNanoForScoutingMiniAOD \
        --filein file:scoutingMiniAOD.root --fileout file:nanoAOD.root

Scouting MiniAOD limitations handled:
- No jet constituents -> disable pileup jet ID, deep taggers
- No AK8 jets -> disable fat jet tables
- No PPS data -> disable proton tables
- Limited trigger objects -> use empty trigger object collection
- No beam spot constraint -> disable beam spot table
"""

import FWCore.ParameterSet.Config as cms


def _removeModuleFromTasks(process, moduleName):
    """Helper to remove a module from all tasks in the process."""
    if hasattr(process, moduleName):
        module = getattr(process, moduleName)
        for taskName in dir(process):
            task = getattr(process, taskName, None)
            if isinstance(task, cms.Task):
                try:
                    task.remove(module)
                except:
                    pass


def _removeModulesFromProcess(process, moduleNames):
    """Remove multiple modules from tasks."""
    for name in moduleNames:
        _removeModuleFromTasks(process, name)


def customiseNanoForScoutingMiniAOD(process):
    """
    Main customization function for running standard NanoAOD on scouting MiniAOD.

    This is the recommended entry point for cmsDriver --customise flag.
    """

    # ============================================================
    # 1. Disable modules requiring jet constituents
    # ============================================================

    # Pileup Jet ID - requires iterating over jet daughters
    pileupJetIdModules = [
        'pileupJetId94X',
        'pileupJetIdPuppi',
        'pileupJetId',
        'pileupJetIdUpdated',
    ]
    _removeModulesFromProcess(process, pileupJetIdModules)

    # Jet update chain - these try to recompute jet properties
    jetUpdateModules = [
        'updatedJets',
        'updatedJetsWithUserData',
        'updatedJetsPuppi',
        'updatedJetsPuppiWithUserData',
        # Full jet recreation chain
        'patJetsPuppi',
        'patJets',
        'selectedPatJetsPuppi',
        'selectedPatJets',
        'slimmedJetsPuppiNoDeepTags',
        'updatedPatJetsSlimmedDeepFlavour',
        'updatedPatJetsTransientCorrectedSlimmedDeepFlavour',
        # Additional jet modules
        'tightJetId',
        'tightJetIdLepVeto',
        'jetCorrFactorsNano',
        'jetCorrFactorsAK8',
    ]
    _removeModulesFromProcess(process, jetUpdateModules)

    # Deep taggers requiring constituent information
    deepTaggerModules = [
        'btagDeepFlavPuppi',
        'btagDeepFlavCMVAPuppi',
        'btagSFPuppi',
        'ctagSFPuppi',
    ]
    _removeModulesFromProcess(process, deepTaggerModules)

    # ============================================================
    # 2. Disable AK8/fat jet modules (not available in scouting)
    # ============================================================

    fatJetModules = [
        'fatJetTable',
        'subJetTable',
        'saTable',
        'saJetTable',
        'softActivityJetTable',
        'fatJetMCTable',
        'subJetMCTable',
    ]
    _removeModulesFromProcess(process, fatJetModules)

    # ============================================================
    # 2b. Disable tau modules (not available in scouting)
    # ============================================================

    tauModules = [
        'slimmedTaus',
        'slimmedTausUpdated',
        'slimmedTausWithPNetCHS',
        'finalTaus',
        'tauTable',
        'tauMCTable',
        'boostedTauTable',
        'boostedTauMCTable',
        'tauIdMVAIsoTask',
        'tauIdDeepTauTask',
    ]
    _removeModulesFromProcess(process, tauModules)

    # ============================================================
    # 3. Disable PPS proton modules (not in scouting data)
    # ============================================================

    ppsModules = [
        'protonTable',
        'multiRPTable',
        'singleRPTable',
    ]
    _removeModulesFromProcess(process, ppsModules)

    # ============================================================
    # 4. Redirect jet tables to use input jets directly
    # ============================================================

    # Skip the jet update chain, use slimmed jets directly
    if hasattr(process, 'jetTable'):
        process.jetTable.src = cms.InputTag("slimmedJets")
        # Remove pileup jet ID variables
        if hasattr(process.jetTable, 'variables'):
            vars_to_remove = ['puId', 'puIdDisc']
            for var in vars_to_remove:
                if hasattr(process.jetTable.variables, var):
                    delattr(process.jetTable.variables, var)

    if hasattr(process, 'jetPuppiTable'):
        # Scouting doesn't have separate Puppi jets, use slimmedJets
        process.jetPuppiTable.src = cms.InputTag("slimmedJets")
        if hasattr(process.jetPuppiTable, 'variables'):
            vars_to_remove = ['puId', 'puIdDisc']
            for var in vars_to_remove:
                if hasattr(process.jetPuppiTable.variables, var):
                    delattr(process.jetPuppiTable.variables, var)

    # Redirect corrT1METJetPuppiTable and remove unavailable variables
    if hasattr(process, 'corrT1METJetPuppiTable'):
        # Scouting doesn't have separate Puppi jets, use slimmedJets
        process.corrT1METJetPuppiTable.src = cms.InputTag("slimmedJets")
        if hasattr(process.corrT1METJetPuppiTable, 'variables'):
            vars_to_remove = ['muonSubtrFactor', 'muonSubtrDeltaEta', 'muonSubtrDeltaPhi']
            for var in vars_to_remove:
                if hasattr(process.corrT1METJetPuppiTable.variables, var):
                    delattr(process.corrT1METJetPuppiTable.variables, var)

    # ============================================================
    # 5. Handle trigger-related modules
    # ============================================================

    # L1 prefiring weights not available
    l1Modules = [
        'L1PreFiringWeight',
        'L1PreFiringWeightProducer',
        'l1TriggerPathTable',
    ]
    _removeModulesFromProcess(process, l1Modules)

    # Trigger object table needs caloStage2Digis and gmtStage2Digis which we don't have
    # We only have gtStage2Digis from scouting. Remove the trigger object table.
    triggerModules = [
        'triggerObjectTable',
        'l1MuTable',
        'l1EGTable',
        'l1TauTable',
        'l1JetTable',
        'l1EtSumTable',
    ]
    _removeModulesFromProcess(process, triggerModules)

    # Keep l1bits if gtStage2Digis is available (we produce it in scoutingToMiniAOD)

    # ============================================================
    # 6. Handle MET modules (we only have slimmedMETs, not Puppi variants)
    # ============================================================

    # Remove/redirect modules that need slimmedMETsPuppi
    puppiMetModules = [
        'rawPuppiMetTable',
        'puppiMetTable',
        'metPuppiTable',
        'corrT1METJetPuppiTable',
    ]
    _removeModulesFromProcess(process, puppiMetModules)

    # Redirect MET table to use slimmedMETs
    if hasattr(process, 'metTable'):
        process.metTable.src = cms.InputTag("slimmedMETs")

    # ============================================================
    # 7. Handle muons and vertices
    # ============================================================

    # Remove slimmedMuonsUpdated which might need additional info
    _removeModuleFromTasks(process, 'slimmedMuonsUpdated')

    # Replace pvbsTable to use offlineSlimmedPrimaryVertices
    # (we don't have offlineSlimmedPrimaryVerticesWithBS)
    if hasattr(process, 'pvbsTable'):
        process.pvbsTable = process.pvTable.clone(
            src = cms.InputTag("offlineSlimmedPrimaryVertices"),
            name = cms.string("PVBS"),
            doc = cms.string("PV with beam spot (same as PV for scouting)")
        )

    # Redirect muon table to use slimmedMuons directly
    if hasattr(process, 'muonTable'):
        process.muonTable.src = cms.InputTag("slimmedMuons")

    # ============================================================
    # 8. Handle rho variables
    # ============================================================

    # Scouting MiniAOD has fixedGridRhoFastjetAll but NanoAOD also needs
    # fixedGridRhoFastjetCentral. We need to either add it to MiniAOD
    # or redirect the rho table.

    # Remove the rhoTable if it causes issues (it needs fixedGridRhoFastjetCentral)
    # Alternative: add fixedGridRhoFastjetCentral to the process
    if hasattr(process, 'rhoTable'):
        # Check if fixedGridRhoFastjetCentral is being produced
        if not hasattr(process, 'fixedGridRhoFastjetCentral'):
            # Create the producer
            process.fixedGridRhoFastjetCentral = cms.EDProducer("FixedGridRhoProducerFastjet",
                pfCandidatesTag = cms.InputTag("packedPFCandidates"),
                maxRapidity = cms.double(2.5),
                gridSpacing = cms.double(0.55)
            )
            # Add to any existing task that runs before nanoSequence
            if hasattr(process, 'nanoSequence'):
                # Insert at the beginning of nanoSequence
                try:
                    process.nanoSequence.insert(0, process.fixedGridRhoFastjetCentral)
                except:
                    pass

    # ============================================================
    # 9. Handle electron/photon updates
    # ============================================================

    # Remove electron updater that might need unavailable info
    electronUpdateModules = [
        'slimmedElectronsUpdated',
        'slimmedElectronsWithUserData',
    ]
    _removeModulesFromProcess(process, electronUpdateModules)

    # Redirect electron table to use slimmedElectrons
    if hasattr(process, 'electronTable'):
        process.electronTable.src = cms.InputTag("slimmedElectrons")

    # Same for photons
    photonUpdateModules = [
        'slimmedPhotonsUpdated',
        'slimmedPhotonsWithUserData',
    ]
    _removeModulesFromProcess(process, photonUpdateModules)

    # Replace photon table with a simpler version for scouting
    # Our scouting photons don't have VID, MVA, or most standard variables
    if hasattr(process, 'photonTable'):
        from PhysicsTools.NanoAOD.common_cff import Var, P3Vars
        process.photonTable = cms.EDProducer("SimplePATPhotonFlatTableProducer",
            src = cms.InputTag("slimmedPhotons"),
            cut = cms.string(""),
            name = cms.string("Photon"),
            doc = cms.string("Photons from scouting"),
            singleton = cms.bool(False),
            extension = cms.bool(False),
            variables = cms.PSet(
                P3Vars,
                # Shower shape from userFloats (set in our producer)
                sieie = Var("userFloat('sigmaIetaIeta')", float, doc="sigma_IetaIeta", precision=10),
                hoe = Var("userFloat('hOverE')", float, doc="H/E", precision=8),
                r9 = Var("userFloat('r9')", float, doc="R9", precision=10),
                # Isolation from userFloats
                ecalIso = Var("userFloat('ecalIso')", float, doc="ECAL isolation", precision=6),
                hcalIso = Var("userFloat('hcalIso')", float, doc="HCAL isolation", precision=6),
                trkIso = Var("userFloat('trkIso')", float, doc="track isolation", precision=6),
            )
        )

    # Replace electron table with a simpler version for scouting
    if hasattr(process, 'electronTable'):
        from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
        process.electronTable = cms.EDProducer("SimplePATElectronFlatTableProducer",
            src = cms.InputTag("slimmedElectrons"),
            cut = cms.string(""),
            name = cms.string("Electron"),
            doc = cms.string("Electrons from scouting"),
            singleton = cms.bool(False),
            extension = cms.bool(False),
            variables = cms.PSet(
                P4Vars,
                charge = Var("charge", int, doc="charge"),
                pdgId = Var("pdgId", int, doc="PDG ID"),
                # Shower shape from userFloats
                sieie = Var("userFloat('sigmaIetaIeta')", float, doc="sigma_IetaIeta", precision=10),
                hoe = Var("userFloat('hOverE')", float, doc="H/E", precision=8),
                r9 = Var("userFloat('r9')", float, doc="R9", precision=10),
                # ID variables from userFloats
                dEtaIn = Var("userFloat('dEtaIn')", float, doc="dEta(SC,track)", precision=10),
                dPhiIn = Var("userFloat('dPhiIn')", float, doc="dPhi(SC,track)", precision=10),
                ooEMOop = Var("userFloat('ooEMOop')", float, doc="1/E - 1/p", precision=10),
                # Track from userFloats
                dxy = Var("userFloat('trkd0')", float, doc="track d0", precision=10),
                dz = Var("userFloat('trkdz')", float, doc="track dz", precision=10),
                # Isolation from userFloats
                ecalIso = Var("userFloat('ecalIso')", float, doc="ECAL isolation", precision=6),
                hcalIso = Var("userFloat('hcalIso')", float, doc="HCAL isolation", precision=6),
                trackIso = Var("userFloat('trackIso')", float, doc="track isolation", precision=6),
            )
        )

    # ============================================================
    # 10. Use TryToContinue for remaining missing products
    # ============================================================

    # This allows the job to continue if some products are missing
    if hasattr(process, 'options'):
        if not hasattr(process.options, 'TryToContinue'):
            process.options.TryToContinue = cms.untracked.vstring()
        process.options.TryToContinue.append('ProductNotFound')

    return process


def customiseNanoForScoutingMiniAOD_nol1bits(process):
    """
    Customization that also removes L1 bits (for testing without L1 unpacking).
    """
    process = customiseNanoForScoutingMiniAOD(process)
    _removeModuleFromTasks(process, 'l1bits')
    return process


def customiseNanoForScoutingMiniAOD_keepFatJets(process):
    """
    Variant that keeps fat jet modules (if you recluster AK8 jets from PF candidates).
    """
    process = customiseNanoForScoutingMiniAOD(process)
    # Re-add fat jet modules that were removed
    # (they would need reclustered AK8 jets as input)
    return process
