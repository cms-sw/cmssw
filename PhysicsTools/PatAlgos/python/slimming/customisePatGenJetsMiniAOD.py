"""Opt-in MiniAOD customisation for pat::GenJet. Nothing here runs, and no pat::GenJet module is added to a process,
unless the function is called (cmsDriver: --customise PhysicsTools/PatAlgos/slimming/customisePatGenJetsMiniAOD.customisePatGenJetsMiniAOD).

    customisePatGenJetsMiniAOD(process, keepInOutput=False)
        adds patGenJetsTask (patGenJetsAK4, packedPatGenJetsAK8 and their helper modules) to the PAT tools
        task, and points the forward links of the pat::GenJets to MiniAOD collections: genJetFwdRef() to the slimmed gen
        jets (slimmedGenJets, slimmedGenJetsAK8, slimmedGenJetsAK8SoftDropSubJets) and, if embedGenCandidates is
        switched on, the stored generator particles to prunedGenParticles. A jet or particle without partner (below the
        slimming cut, dropped by the pruning) keeps its link to the AOD collection, which MiniAOD does not store. With keepInOutput the two products are written to the MiniAOD output.

    customisePatGenJetsMiniAODSupplement(process) / customisePatGenJetsMiniAODReplace(process)
        pat::GenJet as a replacement of the slimmed gen jets. With cmsDriver use --customise_commands (it runs after
        the automatic miniAOD_customizeAllMC, which creates the modules modified here; --customise runs before it):
          --customise_commands "from PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD import \\
            customisePatGenJetsMiniAODReplace; process = customisePatGenJetsMiniAODReplace(process)"
        - the pat::GenJet inputs and cuts follow the live slimmedGenJets* modules (so era and process modifiers such
          as pp_on_AA and genJetSubEvent are followed), and the constituents are kept exactly as PATGenJetSlimmer does
          (key order, a null daughter for a constituent without packed partner);
        - every PATJetSlimmer (and slimmedJets* PATJetSelector) links its jets to their pat::GenJet
          (pat::Jet::patGenJetRef(), used by genJet());
        - patGenJetsAK4FlavourInfos: the gen-jet flavour infos keyed to patGenJetsAK4 (if the process makes
          slimmedGenJetsFlavourInfos);
        - the pat::GenJet products are written to the MiniAOD output.
        Supplement keeps the old collections in the output (for validation); Replace drops slimmedGenJets,
        slimmedGenJetsAK8, slimmedGenJetsAK8SoftDropSubJets and slimmedGenJetsFlavourInfos from the output (their
        producers still run in the job: the reco-jet matching uses them). A job that reads a Replace MiniAOD
        restores the old collections by label with customiseRestoreSlimmedGenJets (below).
        On data (no slimmedGenJets in the process) both are no-ops. Without the AK8 soft-drop generator jets (heavy-ion
        MiniAOD, pp_on_AA: no gen-jet substructure there) only slimmedGenJets (and its flavour infos, if made) is
        replaced; slimmedGenJetsAK8 stays as it is.

    customiseRestoreSlimmedGenJets(process, ak8=True)
        for jobs that READ a Replace MiniAOD (ak8=False: an AK4-only one, e.g. heavy ions): rebuilds slimmedGenJets, slimmedGenJetsAK8,
        slimmedGenJetsAK8SoftDropSubJets (vector<reco::GenJet>) and slimmedGenJetsFlavourInfos from the pat::GenJet
        products, under the old labels, so readers that select them by label need no change. Stored references to the
        old collections (pat::Jet::genJetFwdRef, flavour-info keys) do not point to rebuilt products: pat::Jet links
        through patGenJetRef(), and the NanoAOD expressions are rewritten by customisePatGenJetsNanoReplace.

The NanoAOD tables of these products are added by PhysicsTools/NanoAOD/python/customisePatGenJetsNano.py.
"""
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

# what MiniAOD makes and the pat::GenJet chain reads (the AOD gen jets ak4GenJetsNoNu, ak8GenJetsNoNu come from the input file)
_MINIAOD_INPUTS = ('ak8GenJetsNoNuSoftDrop', 'packedGenParticles', 'prunedGenParticlesWithStatusOne', 'prunedGenParticles',
                   'slimmedGenJets', 'slimmedGenJetsAK8', 'slimmedGenJetsAK8SoftDropSubJets')
# forward link of genJetFwdRef: Association of the slimmer, from the AOD gen jets to the slimmed ones
_GENJET_FORWARD = {'patGenJetsAK4': 'slimmedGenJets', 'patGenJetsAK8': 'slimmedGenJetsAK8',
                   'patGenJetsAK8SoftDropSubjets': 'slimmedGenJetsAK8SoftDropSubJets'}
# forward Ptrs of the generator particles: genParticles -> prunedGenParticlesWithStatusOne -> prunedGenParticles
_GENCANDIDATES_FORWARD = ('prunedGenParticlesWithStatusOne', 'prunedGenParticles')
_KEEP = ('keep *_patGenJetsAK4_*_*', 'keep *_packedPatGenJetsAK8_*_*')


def customisePatGenJetsMiniAOD(process, keepInOutput=False):
    missing = [n for n in _MINIAOD_INPUTS if not hasattr(process, n)]
    if missing:
        raise RuntimeError("customisePatGenJetsMiniAOD: the process lacks %s (MiniAOD makes them); the pat::GenJet chain needs them" % missing)
    process.load('PhysicsTools.PatAlgos.producersLayer1.patGenJets_cff')
    getPatAlgosToolsTask(process).add(process.patGenJetsTask)
    for label, slimmed in _GENJET_FORWARD.items():
        producer = getattr(process, label)
        producer.genJetForwardMap = cms.InputTag(slimmed, 'slimmedGenJetAssociation')
        producer.genCandidatesForwardMaps = cms.VInputTag(*_GENCANDIDATES_FORWARD)
    if keepInOutput:
        done = 0
        for label, out in process.outputModules_().items():
            commands = out.outputCommands
            if not any('slimmedGenJets' in c for c in commands):
                continue  # not a MiniAOD output module
            for k in _KEEP:
                if k not in commands:
                    commands.append(k)
            done += 1
        if done == 0:
            raise RuntimeError("customisePatGenJetsMiniAOD: keepInOutput=True but no output module keeps slimmedGenJets")
    return process



# ---------------------------------------------------------------------------------------------------------------------
# pat::GenJet as a replacement of the slimmed gen jets

_OLD_GENJETS = ('slimmedGenJets', 'slimmedGenJetsAK8', 'slimmedGenJetsAK8SoftDropSubJets')
# patGenJetsAK4: only the jets (its Association, input of the flavour infos in this job, is not needed downstream)
_PAT_PRODUCTS = ('keep patGenJets_patGenJetsAK4__*', 'keep *_packedPatGenJetsAK8_*_*', 'keep *_patGenJetsAK4FlavourInfos_*_*')


def _follow_slimmed_gen_jets(process, ak8chain):
    """pat::GenJet inputs and selection = those of the live slimmedGenJets* modules (after all modifiers)."""
    def same(a, b):
        return cms.InputTag(a).value() == cms.InputTag(b).value()
    ak4 = process.slimmedGenJets
    if not ak8chain:
        if ak4.cut.value() != '' or ak4.cutLoose.value() != '' or ak4.nLoose.value() != 0 or ak4.clearDaughters.value() \
           or ak4.dropSpecific.value():
            raise RuntimeError("customisePatGenJetsMiniAOD: slimmedGenJets settings not reproduced by the AK4 chain")
        process.patGenJetsAK4.jetSource = ak4.src.value()
        process.patGenJetsAK4.daughterOrder = 'key'
        process.patGenJetsAK4.keepUnmappedDaughters = True
        if not same(process.patGenJetsAK4.packedGenParticles.value(), ak4.packedGenParticles.value()):
            raise RuntimeError("customisePatGenJetsMiniAOD: patGenJetsAK4 and slimmedGenJets use different packedGenParticles")
        return
    ak4, ak8, sub = (getattr(process, n) for n in _OLD_GENJETS)
    # AK4: no selection in the pat::GenJet chain
    if ak4.cut.value() != '' or ak4.cutLoose.value() != '' or ak4.nLoose.value() != 0:
        raise RuntimeError("customisePatGenJetsMiniAOD: slimmedGenJets has a selection (cut '%s', nLoose %d) that the "
                           "pat::GenJet AK4 chain does not reproduce" % (ak4.cut.value(), ak4.nLoose.value()))
    process.patGenJetsAK4.jetSource = ak4.src.value()
    # AK8: the substructure modules read ak8GenJetsNoNu; a different source is not followed
    if not same(ak8.src.value(), 'ak8GenJetsNoNu'):
        raise RuntimeError("customisePatGenJetsMiniAOD: slimmedGenJetsAK8 reads %s; the pat::GenJet AK8 substructure "
                           "reads ak8GenJetsNoNu" % ak8.src.value())
    process.selectedPatGenJetsAK8.cut = ak8.cut.value()
    process.selectedPatGenJetsAK8.cutLoose = ak8.cutLoose.value()
    process.selectedPatGenJetsAK8.nLoose = ak8.nLoose.value()
    process.patGenJetsAK8SoftDropSubjets.jetSource = sub.src.value()
    # the slimmers' constituent and Specific settings that the replacement reproduces
    for m in (ak4, ak8, sub):
        if m.clearDaughters.value():
            raise RuntimeError("customisePatGenJetsMiniAOD: a slimmed gen-jet collection clears its daughters; not reproduced")
    if ak4.dropSpecific.value() or ak8.dropSpecific.value() or not sub.dropSpecific.value():
        raise RuntimeError("customisePatGenJetsMiniAOD: unexpected dropSpecific settings of the slimmed gen jets; not reproduced")
    # constituents exactly as PATGenJetSlimmer
    for label in _GENJET_FORWARD:
        getattr(process, label).daughterOrder = 'key'
        getattr(process, label).keepUnmappedDaughters = True
    for label, old in zip(('patGenJetsAK4', 'patGenJetsAK8', 'patGenJetsAK8SoftDropSubjets'), _OLD_GENJETS):
        if not same(getattr(process, label).packedGenParticles.value(), getattr(process, old).packedGenParticles.value()):
            raise RuntimeError("customisePatGenJetsMiniAOD: %s and %s use different packedGenParticles" % (label, old))


_AK4_INPUTS = ('packedGenParticles', 'prunedGenParticlesWithStatusOne', 'prunedGenParticles', 'slimmedGenJets')


def _replacement(process, dropOld):
    if not hasattr(process, 'slimmedGenJets'):
        print("customisePatGenJetsMiniAOD: no slimmed gen jets in the process: nothing changed (data; or called before "
              "miniAOD_customizeAllMC - with cmsDriver use --customise_commands, which runs after it)")
        return process
    ak8chain = all(hasattr(process, n) for n in _MINIAOD_INPUTS)
    task = getPatAlgosToolsTask(process)
    if ak8chain:
        process = customisePatGenJetsMiniAOD(process, keepInOutput=False)
        replaced = _OLD_GENJETS
        patGenJets = cms.VInputTag('patGenJetsAK4', 'packedPatGenJetsAK8', cms.InputTag('packedPatGenJetsAK8', 'SubJets'))
        keep = _PAT_PRODUCTS
    else:
        missing = [n for n in _AK4_INPUTS if not hasattr(process, n)]
        if missing:
            raise RuntimeError("customisePatGenJetsMiniAOD: the process lacks %s" % missing)
        # AK4 only (e.g. heavy-ion MiniAOD: no AK8 soft-drop generator jets)
        process.load('PhysicsTools.PatAlgos.producersLayer1.patGenJets_cff')
        task.add(process.patGenJetsAK4)
        process.patGenJetsAK4.genJetForwardMap = cms.InputTag('slimmedGenJets', 'slimmedGenJetAssociation')
        process.patGenJetsAK4.genCandidatesForwardMaps = cms.VInputTag(*_GENCANDIDATES_FORWARD)
        replaced = ('slimmedGenJets',)
        patGenJets = cms.VInputTag('patGenJetsAK4')
        keep = ('keep patGenJets_patGenJetsAK4__*', 'keep *_patGenJetsAK4FlavourInfos_*_*')
        print("customisePatGenJetsMiniAOD: no AK8 soft-drop generator jets in the process: only slimmedGenJets is replaced")
    _follow_slimmed_gen_jets(process, ak8chain)
    # reco jets -> pat::GenJet: every PATJetSlimmer, and every PATJetSelector that writes a slimmedJets* collection
    # (heavy-ion MiniAOD has no slimmers), searches the pat::GenJet collections for the twin of its jets' matched gen
    # jet (jets without matched gen jet, or with one outside these collections, get no link)
    nslim = 0
    for label, module in process.producers_().items():
        if module.type_() == 'PATJetSlimmer':
            module.patGenJets = patGenJets
            nslim += 1
    for label, module in process.filters_().items():
        if module.type_() == 'PATJetSelector' and label.startswith('slimmedJets'):
            module.patGenJets = patGenJets
            nslim += 1
    if nslim == 0:
        raise RuntimeError("customisePatGenJetsMiniAOD: no PATJetSlimmer or slimmedJets* PATJetSelector in the process")
    # gen-jet flavour infos keyed to patGenJetsAK4 (the same inputs as slimmedGenJetsFlavourInfos)
    old_flavour = getattr(process, 'slimmedGenJetsFlavourInfos', None)
    if old_flavour is not None and task.contains(old_flavour):
        process.patGenJetsAK4.produceAssociation = True
        process.patGenJetsAK4FlavourInfos = cms.EDProducer('PATGenJetFlavourInfoPreserver',
            genJets=old_flavour.genJets,
            slimmedGenJets=cms.InputTag('patGenJetsAK4'),
            genJetFlavourInfos=old_flavour.genJetFlavourInfos,
            slimmedGenJetAssociation=cms.InputTag('patGenJetsAK4'))
        task.add(process.patGenJetsAK4FlavourInfos)
        replaced = replaced + ('slimmedGenJetsFlavourInfos',)
    # output
    done = 0
    for label, out in process.outputModules_().items():
        commands = out.outputCommands
        if not any('slimmedGenJets' in c for c in commands):
            continue  # not a MiniAOD output module
        for k in keep + (tuple('drop *_%s_*_*' % n for n in replaced) if dropOld else ()):
            if k not in commands:
                commands.append(k)
        done += 1
    if done == 0:
        raise RuntimeError("customisePatGenJetsMiniAOD: no output module keeps slimmedGenJets")
    print("customisePatGenJetsMiniAOD: %s: pat::GenJet %s %s; %d jet slimmers/selectors linked" % (
        'Replace' if dropOld else 'Supplement', 'replaces' if dropOld else 'stored beside', ', '.join(replaced), nslim))
    return process


def customisePatGenJetsMiniAODSupplement(process):
    return _replacement(process, dropOld=False)


def customisePatGenJetsMiniAODReplace(process):
    return _replacement(process, dropOld=True)


def customiseRestoreSlimmedGenJets(process, ak8=True):
    """Reading job of a Replace MiniAOD: old gen-jet collections rebuilt by label from the pat::GenJet products."""
    restored = (_OLD_GENJETS if ak8 else ('slimmedGenJets',)) + ('slimmedGenJetsFlavourInfos',)
    taken = [n for n in restored if hasattr(process, n)]
    if taken:
        raise RuntimeError("customiseRestoreSlimmedGenJets: the process already defines %s (a MiniAOD-making job?); "
                           "the restore is for jobs that read MiniAOD" % taken)
    process.slimmedGenJets = cms.EDProducer('PATGenJetToRecoGenJets',
        src=cms.InputTag('patGenJetsAK4'), dropSpecific=cms.bool(False))
    if ak8:
        process.slimmedGenJetsAK8 = cms.EDProducer('PATGenJetToRecoGenJets',
            src=cms.InputTag('packedPatGenJetsAK8'), dropSpecific=cms.bool(False))
        # PATGenJetSlimmer of the soft-drop subjets drops the Specific block (applySubstructure_cff.py)
        process.slimmedGenJetsAK8SoftDropSubJets = cms.EDProducer('PATGenJetToRecoGenJets',
            src=cms.InputTag('packedPatGenJetsAK8', 'SubJets'), dropSpecific=cms.bool(True))
    # flavour infos re-keyed from patGenJetsAK4 to the rebuilt slimmedGenJets (same jets, same order)
    process.slimmedGenJetsFlavourInfos = cms.EDProducer('GenJetFlavourInfoPreserver',
        genJets=cms.InputTag('patGenJetsAK4'),
        slimmedGenJets=cms.InputTag('slimmedGenJets'),
        genJetFlavourInfos=cms.InputTag('patGenJetsAK4FlavourInfos'),
        slimmedGenJetAssociation=cms.InputTag('slimmedGenJets'))
    process.restoreSlimmedGenJetsTask = cms.Task(*[getattr(process, n) for n in restored])
    if hasattr(process, 'schedule') and process.schedule is not None:
        process.schedule.associate(process.restoreSlimmedGenJetsTask)
    else:
        raise RuntimeError("customiseRestoreSlimmedGenJets: the process has no schedule to attach the task to")
    return process
