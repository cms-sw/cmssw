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
                beamSpot=cms.InputTag("hltOnlineBeamSpot"),
                ptMin = cms.double( 1.5 ),
                originHalfLength = cms.double( 12.5 ),
                originRadius = cms.double( 0.05 )
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


def customizeHLTfor51085(process):
    """ This customizer
        - adds some of the geometry parameters used to fill the CAGeometry:
            - minZ -> minInner
            - maxZ -> maxInner
            - maxR -> maxDR
        - for pp and HIN hlt setups.
    """
 
    ca_producers_pp = ['CAHitNtupletAlpakaPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaPhase1']
    ca_producers_hi = ['CAHitNtupletAlpakaHIonPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaHIonPhase1']
    ca_producers = ca_producers_pp + ca_producers_hi

    for ca_producer in ca_producers:
        for prod in producers_by_type(process, ca_producer):

            # get the current geometry
            geometry = getattr(prod, "geometry")

            nLayers = len(geometry.caDCACuts)
            nPairs = len(geometry.maxDR)
            
            # new geometry parameters
            if not hasattr(geometry, "startMaxInnerR"):
                setattr(geometry, "startMaxInnerR", cms.vdouble([1000.] * nLayers))
                
            if not hasattr(geometry, "maxDCurv"):
                setattr(geometry, "maxDCurv", cms.vdouble([1000.] * nLayers))
                
            if not hasattr(geometry, "floorDCurv"):
                setattr(geometry, "floorDCurv", cms.vdouble([1000.] * nLayers))
                
            if not hasattr(geometry, "fishboneCuts"):
                setattr(geometry, "fishboneCuts", cms.vdouble([0.99999] * nLayers))
                
            if not hasattr(geometry, "skipsLayers"):
                setattr(geometry, "skipsLayers", cms.vuint32([False] * nPairs))

            # set the full geometry
            setattr(prod, 'geometry', geometry)

    return process


def customizeHLTfor52006(process):
    """ This customizer
        - adds the ONNXService, required by all the modules that use the ONNX Runtime,
          if the menu contains any of them.
    """

    onnx_modules = [
        'BoostedJetONNXJetTagsProducer',
        'DeepCombinedONNXJetTagsProducer',
        'DeepDoubleXONNXJetTagsProducer',
        'DeepFlavourONNXJetTagsProducer',
        'DeepVertexONNXJetTagsProducer',
        'HLTParticleTransformerAK4ONNXJetTagsProducer',
        'ParticleTransformerAK4ONNXJetTagsProducer',
        'UnifiedParticleTransformerAK4ONNXJetTagsProducer',
    ]

    if not hasattr(process, 'ONNXService') and next(producers_by_type(process, *onnx_modules), None) is not None:
        process.ONNXService = cms.Service('ONNXService')

    return process


def customizeHLTforL1P2GTv0p0_2026(process):
    """ This customizer
        - re-wires the L1 P2GT seeds of the Phase-2 HLT menu (HLT:75e33), written for the
          step1_2024 L1 menu, to the v0p0_2026 L1 menu (introduced in #52372), to run with
          -s L1P2GT:v0p0_2026,HLT:75e33
        - it is NOT part of customizeHLTforCMSSW: it has to be requested explicitly via --customise
        All the step1_2024 seeds are mapped, including those not (yet) used by the HLT menu,
        to their v0p0_2026 counterpart (pDoubleDispMu, pDoublePFJet8_dEtaMax and pSinglePfJet8
        keep their name):
        - same / retuned: same objects and multiplicity, new thresholds
        - closest: nearest analogue with a different topology, the objects a filter
          expects from the seed may need to be adapted when it starts using it
        Seeds without a v0p0_2026 counterpart are dropped from the L1 filters that use them:
        - pIsoTkEleEGEle22_12: dropped from the OR of the isolated double-electron L1 filter
        - pPuppiHT400_pQuadJet70_55_40_40: no quad-jet algo in v0p0_2026, the
          HT + quad-jet + 3b paths fall back to pHTSum alone
    """

    seedMap = {
        # same
        'pSingleEGEle51'                                 : 'pSingleEGEle',
        # retuned
        'pDoubleEGEle37_24'                              : 'pDoubleEGEle',
        'pDoubleIsoTkPho22_12'                           : 'pDoubleTkPhoIso',
        'pDoublePuppiJet112_112'                         : 'pDoubleJet_dEta_Max1p6',
        'pDoublePuppiTau52_52'                           : 'pDoubleIsoTau',
        'pDoubleTkEle25_12'                              : 'pDoubleTkEle',
        'pDoubleTkElePuppiHT_8_8_390'                    : 'pDoubleTkEle_HTSum',
        'pDoubleTkMuPuppiHT_3_3_300'                     : 'pDoubleTkMu_HTSum',
        'pDoubleTkMuon15_7'                              : 'pDoubleTkMu',
        'pDoubleTkMuonTkEle5_5_9'                        : 'pDoubleTkMu_TkEle',
        'pDoubleTkMuon_OS_Er1p5_Dr1p4'                   : 'pDoubleTkMu0er1p4_SQ_OS_dR_Max1p4',
        'pPuppiHT400'                                    : 'pHTSum',
        'pPuppiHT450'                                    : 'pHTSum',
        'pPuppiMET200'                                   : 'pMETSum',
        'pPuppiTauTkIsoEle45_22'                         : 'pTkEleIso_IsoTau',
        'pPuppiTauTkMuon42_18'                           : 'pTkMu_IsoTau',
        'pSingleIsoTkEle28'                              : 'pSingleTkEleIso',
        'pSingleIsoTkPho36'                              : 'pSingleTkIsoPhoton',
        'pSinglePuppiJet230'                             : 'pSingleJet',
        'pSingleTkEle36'                                 : 'pSingleTkEle',
        'pSingleTkMuon22'                                : 'pSingleTkMu',
        'pTkEleIsoPuppiHT_26_190'                        : 'pTkEleIso_HTSum',
        'pTkEleTkMuon10_20'                              : 'pTkMu20_TkEle10',
        'pTkMuPuppiJetPuppiMet_3_110_120'                : 'pTkMu_Jet_METSum',
        'pTkMuonDoubleTkEle6_17_17'                      : 'pTkMu_DoubleTkEle',
        'pTkMuonPuppiHT6_320'                            : 'pTkMu6_HTSum240',
        'pTkMuonTkEle7_23'                               : 'pTkMu7_TkEle20',
        'pTripleTkMuon5_3_3'                             : 'pTripleTkMu',
        # closest
        'pDoublePuppiJet160_35_mass620'                  : 'pDoubleJet_Mass_Min1000',
        'pDoubleTkMuPuppiJetPuppiMet_3_3_60_130'         : 'pDoubleTkMu_METSum_HTSum',
        'pDoubleTkMuon_4_4_OS_Dr1p2'                     : 'pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6',
        'pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18'         : 'pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6',
        'pSingleDispMu'                                  : 'pSingleDispMu_BMTF',
        'pTkMuonTkIsoEle7_20'                            : 'pTkMu7_TkEle20',
        'pTripleTkMuon_5_3_0_DoubleTkMuon_5_3_OS_MassTo9': 'pTripleTkMu_4_2p5_0_OS_Mass_Max12',
        'pTripleTkMuon_5_3p5_2p5_OS_Mass5to17'           : 'pTripleTkMu_4_2p5_0_OS_Mass_Max12',
        # no counterpart, HT leg only (see above)
        'pPuppiHT400_pQuadJet70_55_40_40'                : 'pHTSum',
    }
    # no counterpart
    droppedSeeds = [
        'pIsoTkEleEGEle22_12',
        'pNNPuppiTauPuppiMet_55_190',
        'pPuppiMHT140',
        'pQuadJet70_55_40_40',
        'pTkElePuppiJet_28_40_MinDR',
        'pTkMuTriPuppiJet_12_40_dRMax_DoubleJet_dEtaMax',
    ]

    # drop the quad-jet leg of the HT + quad-jet seed, the HT leg is re-wired to pHTSum below
    if hasattr(process, 'hltL1QuadJet70554040'):
        quadJetFilter = process.hltL1QuadJet70554040
        for seq in [*process.paths_().values(), *process.endpaths_().values(), *process.sequences_().values()]:
            while seq.remove(quadJetFilter):
                pass
        del process.hltL1QuadJet70554040

    for module in process.filters_().values():
        if hasattr(module, 'l1GTAlgos'):
            module.l1GTAlgos = cms.VPSet([pset for pset in module.l1GTAlgos if pset.name.value() not in droppedSeeds])
            for pset in module.l1GTAlgos:
                pset.name = seedMap.get(pset.name.value(), pset.name.value())
        if hasattr(module, 'l1GTAlgoNames'):
            module.l1GTAlgoNames = [seedMap.get(name, name) for name in module.l1GTAlgoNames if name not in droppedSeeds]
        for par in ['l1GTAlgoName1', 'l1GTAlgoName2']:
            if hasattr(module, par):
                name = getattr(module, par).value()
                setattr(module, par, seedMap.get(name, name))

    return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    # process = customizeHLTfor49436(process)
    process = customizeHLTfor51085(process)
    process = customizeHLTfor52006(process)

    return process
