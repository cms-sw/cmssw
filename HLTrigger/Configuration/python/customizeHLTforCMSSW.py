import FWCore.ParameterSet.Config as cms

# helper functions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

def customiseForOffline(process):
    # For running HLT offline and relieve the strain on Frontier so it will no longer inject a
    # transaction id which tells Frontier to add a unique "&freshkey" to many query URLs.
    # That was intended as a feature to only be used by the Online HLT, to guarantee that fresh conditions
    # from the database were loaded at each Lumi section
    # Seee CMSHLT-3123 for further details
    if hasattr(process, 'GlobalTag'):
        # Set ReconnectEachRun and RefreshEachRun to False
        process.GlobalTag.ReconnectEachRun = cms.untracked.bool(False)
        process.GlobalTag.RefreshEachRun = cms.untracked.bool(False)

        if hasattr(process.GlobalTag, 'toGet'):
            # Filter out PSet objects containing only 'record' and 'refreshTime'
            process.GlobalTag.toGet = [
                pset for pset in process.GlobalTag.toGet
                if set(pset.parameterNames_()) != {'record', 'refreshTime'}
            ]

    return process

def replace_all_pixel_seed_inputtags(process):
    import FWCore.ParameterSet.Config as cms

    replacements = {
        "hltEgammaElectronPixelSeeds": "hltEgammaFittedElectronPixelSeeds",
        "hltEgammaElectronPixelSeedsUnseeded": "hltEgammaFittedElectronPixelSeedsUnseeded",
        "hltEgammaElectronPixelSeedsForBParkingUnseeded": "hltEgammaFittedElectronPixelSeedsForBParkingUnseeded",
    }

    InputTag = cms.InputTag
    skip_modules = set(replacements.values())

    def replace_in_module(name, module):
        # ! Skip the module that PRODUCES the product we are renaming
        if name in skip_modules:
            return

        for pName in module.parameters_():
            val = getattr(module, pName)

            # --- Case 1: a single InputTag ---
            if isinstance(val, InputTag):
                old_mod = val.getModuleLabel()
                if old_mod in replacements:
                    setattr(module, pName,
                            cms.InputTag(replacements[old_mod],
                                         val.getProductInstanceLabel(),
                                         val.getProcessName()))
            # --- Case 2: VInputTag / list of InputTag ---
            elif isinstance(val, (cms.VInputTag, list, tuple)):
                new_list = []
                changed = False
                for it in val:
                    if isinstance(it, InputTag) and it.getModuleLabel() in replacements:
                        old_mod = it.getModuleLabel()
                        it = cms.InputTag(replacements[old_mod],
                                          it.getProductInstanceLabel(),
                                          it.getProcessName())
                        changed = True
                    new_list.append(it)
                if changed:
                    setattr(module, pName, type(val)(new_list))

            # --- Case 3: nested PSet ---
            elif hasattr(val, "parameters_"):
                replace_in_pset(val)

    def replace_in_pset(pset):
        for pName in pset.parameters_():
            val = getattr(pset, pName)
            if isinstance(val, InputTag):
                old_mod = val.getModuleLabel()
                if old_mod in replacements:
                    setattr(pset, pName,
                            cms.InputTag(replacements[old_mod],
                                         val.getProductInstanceLabel(),
                                         val.getProcessName()))
            elif isinstance(val, (cms.VInputTag, list, tuple)):
                new_list = []
                changed = False
                for it in val:
                    if isinstance(it, InputTag) and it.getModuleLabel() in replacements:
                        old_mod = it.getModuleLabel()
                        it = cms.InputTag(replacements[old_mod],
                                          it.getProductInstanceLabel(),
                                          it.getProcessName())
                        changed = True
                    new_list.append(it)
                if changed:
                    setattr(pset, pName, type(val)(new_list))
            elif hasattr(val, "parameters_"):
                replace_in_pset(val)

    # Apply to all modules
    for name,mod in process.producers_().items():
        replace_in_module(name,mod)
    for name,mod in process.filters_().items():
        replace_in_module(name,mod)
    for name,mod in process.analyzers_().items():
        replace_in_module(name,mod)

    # also walk top-level PSets
    for pset in process.psets_().values():
        replace_in_pset(pset)

def customizeHLTfor49436(process):

    # Replace Ele Pixel Seeds Doublets/Triplets
    replacements = {
        "hltElePixelSeedsDoublets": ("hltElePixelHitDoublets"),
        "hltElePixelSeedsDoubletsUnseeded": ("hltElePixelHitDoubletsUnseeded"),
        "hltElePixelSeedsTriplets": ("hltElePixelHitTriplets"),
        "hltElePixelSeedsTripletsUnseeded": ("hltElePixelHitTripletsUnseeded"),
    }

    for module_name, hitset in replacements.items():
        if hasattr(process, module_name):
            setattr(
                process,
                module_name,
                cms.EDProducer(
                    "FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer",
                    seedingHitSets=cms.InputTag(hitset),
                    SeedComparitorPSet=cms.PSet(
                        ComponentName=cms.string("none")
                    )
                )
            )

    # Add new ElectronSeedFitter modules
    fitter_configs = {
        "hltEgammaFittedElectronPixelSeeds": "hltEgammaElectronPixelSeeds",
        "hltEgammaFittedElectronPixelSeedsUnseeded": "hltEgammaElectronPixelSeedsUnseeded",
        "hltEgammaFittedElectronPixelSeedsForBParkingUnseeded": "hltEgammaElectronPixelSeedsForBParkingUnseeded",
    }

    for mod_name, input_collection in fitter_configs.items():
        setattr(
            process,
            mod_name,
            cms.EDProducer(
                "ElectronSeedFitter",
                eleSeedCollection=cms.InputTag(input_collection),
                propagator=cms.string("PropagatorWithMaterialParabolicMf"),
                SeedMomentumForBOFF=cms.double(5.0),
                OriginTransverseErrorMultiplier=cms.double(1.0),
                MinOneOverPtError=cms.double(1.0),
                TTRHBuilder=cms.string("hltESPTTRHBWithTrackAngle"),
                magneticField=cms.string("ParabolicMf"),
            )
        )

    # Global replacements of pixel seed producers
    replace_all_pixel_seed_inputtags(process)

    # Insert new modules into the 3 sequences
    # Mapping of sequences -> (new module, pixelMatchVars module)
    seq_updates = [
        ("HLTElePixelMatchSequence",
         "hltEgammaFittedElectronPixelSeeds",
         "hltEgammaPixelMatchVars"),

        ("HLTElePixelMatchUnseededSequence",
         "hltEgammaFittedElectronPixelSeedsUnseeded",
         "hltEgammaPixelMatchVarsUnseeded"),

        ("HLTElePixelMatchUnseededSequenceForBParking",
         "hltEgammaFittedElectronPixelSeedsForBParkingUnseeded",
         "hltEgammaPixelMatchVarsForBParkingUnseeded"),
    ]

    for seq_name, new_mod, match_mod in seq_updates:
        if hasattr(process, seq_name) and hasattr(process, new_mod) and hasattr(process, match_mod):
            seq = getattr(process, seq_name)
            new_module = getattr(process, new_mod)
            match_module = getattr(process, match_mod)
            # Insert the new module immediately before pixelMatchVars
            seq.replace(match_module, new_module + match_module)

    return process

def customizeHLTfor49936(process):
    """
    Update Pixel comparison and monitoring modules to comply with the
    SoA-based plugin migration introduced in PR #49936.

    Existing HLT menus use legacy plugin types (e.g.
    SiPixelPhase1CompareRecHits); this customization updates the C++
    plugin type in place while preserving:
      - module labels
      - parameters
      - path and task scheduling
    """

    from HLTrigger.Configuration.common import producers_by_type

    type_map = {
        "SiPixelPhase1CompareRecHits": "SiPixelCompareRecHitsSoA",
        "SiPixelPhase1CompareTracks": "SiPixelCompareTracksSoA",
        "SiPixelCompareVertices": "SiPixelCompareVerticesSoA",

        "SiPixelPhase1MonitorRecHitsSoAAlpaka": "SiPixelMonitorRecHitsSoA",
        "SiPixelPhase1MonitorTrackSoAAlpaka": "SiPixelMonitorTrackSoA",
        "SiPixelMonitorVertexSoAAlpaka": "SiPixelMonitorVertexSoA",
    }

    for old_type, new_type in type_map.items():
        for module in producers_by_type(process, old_type):
            module._TypedParameterizable__type = new_type

    return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    # process = customizeHLTfor49436(process)

    process = customizeHLTfor49936(process)

    return process
