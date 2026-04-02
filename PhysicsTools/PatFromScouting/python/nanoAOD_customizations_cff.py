"""
Customizations for running NanoAOD on scouting data converted to MiniAOD format.

Scouting data has limited information compared to full reconstruction,
so some NanoAOD modules need to be disabled or modified.
"""

import FWCore.ParameterSet.Config as cms

def customizeNanoAODForScouting(process):
    """
    Customize NanoAOD for scouting data input.

    Disables modules that require information not available in scouting:
    - Pileup jet ID (requires jet constituents)
    - Deep jet taggers (requires jet constituents)
    - b-tag shape corrections
    - AK8 jets (not available in scouting)
    - PPS forward protons (not in scouting)
    """

    # Remove pileup jet ID modules (they crash without jet constituents)
    modulesToRemove = [
        'pileupJetId94X',
        'pileupJetIdPuppi',
        'pileupJetId',
        'pileupJetIdUpdated',
        'updatedJets',
        'updatedJetsWithUserData',
        'updatedJetsPuppi',
        'updatedJetsPuppiWithUserData',
        # AK8 jets (not available in scouting)
        'fatJetTable',
        'subJetTable',
        'saTable',
        'saJetTable',
        # Deep taggers that need constituents
        'btagDeepFlavPuppi',
        'btagDeepFlavCMVAPuppi',
        # B-tag corrections
        'btagSFPuppi',
        'ctagSFPuppi',
    ]

    # Remove modules from tasks
    for moduleName in modulesToRemove:
        if hasattr(process, moduleName):
            module = getattr(process, moduleName)
            for taskName in dir(process):
                task = getattr(process, taskName)
                if isinstance(task, cms.Task):
                    try:
                        task.remove(module)
                    except:
                        pass

    # Alternative approach: modify the slimmedJets producer to skip pileup jet ID
    # by providing empty jet ID information
    if hasattr(process, 'updatedPatJets'):
        process.updatedPatJets.addJetID = cms.bool(False)

    if hasattr(process, 'updatedPatJetsPuppi'):
        process.updatedPatJetsPuppi.addJetID = cms.bool(False)

    # Fix jet tables to not depend on pileup jet ID
    if hasattr(process, 'jetTable'):
        # Remove variables that depend on pileup jet ID
        try:
            del process.jetTable.variables.puId
            del process.jetTable.variables.puIdDisc
        except:
            pass

    if hasattr(process, 'jetPuppiTable'):
        try:
            del process.jetPuppiTable.variables.puId
            del process.jetPuppiTable.variables.puIdDisc
        except:
            pass

    # Disable PPS proton tables (not available in scouting)
    # Remove from output and tasks instead of trying to configure empty sources
    ppsModules = ['protonTable', 'multiRPTable', 'singleRPTable']
    for moduleName in ppsModules:
        if hasattr(process, moduleName):
            module = getattr(process, moduleName)
            # Remove from tasks
            for taskName in dir(process):
                task = getattr(process, taskName)
                if isinstance(task, cms.Task):
                    try:
                        task.remove(module)
                    except:
                        pass
            # Also try removing from sequences
            for seqName in dir(process):
                seq = getattr(process, seqName)
                if isinstance(seq, cms.Sequence):
                    try:
                        seq.remove(module)
                    except:
                        pass

    # Redirect jet modules that depend on updatedJetsPuppiWithUserData
    # to use slimmedJets directly (scouting doesn't have separate Puppi jets)
    if hasattr(process, 'corrT1METJetPuppiTable'):
        process.corrT1METJetPuppiTable.src = cms.InputTag("slimmedJets")
        # Remove variables that need user data
        try:
            del process.corrT1METJetPuppiTable.variables.muonSubtrFactor
            del process.corrT1METJetPuppiTable.variables.muonSubtrDeltaEta
            del process.corrT1METJetPuppiTable.variables.muonSubtrDeltaPhi
        except:
            pass

    # Redirect jetPuppiTable if it exists (scouting doesn't have separate Puppi jets)
    if hasattr(process, 'jetPuppiTable'):
        process.jetPuppiTable.src = cms.InputTag("slimmedJets")

    # Redirect jetTable to use slimmedJets
    if hasattr(process, 'jetTable'):
        process.jetTable.src = cms.InputTag("slimmedJets")

    # Disable modules that need data not available in scouting
    modulesToDisable = [
        'beamSpotTable',           # No beam spot
        'l1bits',                  # L1 trigger bits (gtStage2Digis)
        'L1PreFiringWeight',       # L1 prefiring
        'L1PreFiringWeightProducer',
        # Additional L1/HLT related
        'l1TriggerPathTable',
        'triggerObjectTable',
        'TrigObjFlatTableProducer',
    ]

    for moduleName in modulesToDisable:
        if hasattr(process, moduleName):
            module = getattr(process, moduleName)
            for taskName in dir(process):
                task = getattr(process, taskName)
                if isinstance(task, cms.Task):
                    try:
                        task.remove(module)
                    except:
                        pass

    return process


def customizeNanoAODForScoutingMinimal(process):
    """
    Minimal customization that only disables crashing modules.
    """
    # Simply remove the pileup jet ID producers from all paths and tasks
    toRemove = ['pileupJetIdPuppi', 'pileupJetId', 'pileupJetIdUpdated']

    for name in toRemove:
        if hasattr(process, name):
            delattr(process, name)

    return process
