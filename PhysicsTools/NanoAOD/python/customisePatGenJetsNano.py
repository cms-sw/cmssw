"""Opt-in NanoAOD customisation for pat::GenJet. Nothing here runs, and no pat::GenJet table is added to a process,
unless the function is called (cmsDriver: --customise PhysicsTools/NanoAOD/customisePatGenJetsNano.customisePatGenJetsNano).

    customisePatGenJetsNano(process)
        adds the tables PatGenJet, PatGenJetAK8, PatSubGenJetAK8 to the NanoAOD table task (and the DQM entries).
        Call it AFTER nanoAOD_customizeCommon (cmsDriver appends the customisations in the order given): the column
        sets are copied from the tables of the running process. The tables read the pat::GenJet products: same job as
        customisePatGenJetsMiniAOD (PhysicsTools/PatAlgos/python/slimming/customisePatGenJetsMiniAOD.py), or a MiniAOD
        written with keepInOutput=True.

    customisePatGenJetsNanoReplace(process, ak8=True)
        NanoAOD from a MiniAOD written with customisePatGenJetsMiniAODReplace (old gen-jet collections dropped):
        - rebuilds slimmedGenJets, slimmedGenJetsAK8, slimmedGenJetsAK8SoftDropSubJets and slimmedGenJetsFlavourInfos
          by label (customiseRestoreSlimmedGenJets), so the GenJet, GenJetAK8, SubGenJetAK8 and flavour tables and
          every other reader of these labels run unchanged;
        - rewrites every string expression that follows the stored gen-jet reference of a pat::Jet
          (genJetFwdRef().backRef(), whose product is not in the file) to pat::Jet::matchedGenJetPtr(): the pat::GenJet
          link if the jet has it, else the original back link (jets re-made in this job link to the rebuilt collection).
          The keys are the same, so Jet_genJetIdx, FatJet_genJetAK8Idx and SubJet_subGenJetAK8Idx are unchanged.
        Call it LAST (after nanoAOD_customizeCommon, custom_jme, BTV customisations): it rewrites what is configured.
        With cmsDriver use --customise_commands, which runs after the automatic NanoAOD customisations.
"""
import FWCore.ParameterSet.Config as cms
import re


def customisePatGenJetsNano(process):
    process.load('PhysicsTools.NanoAOD.patGenJetTables_cff')
    from PhysicsTools.NanoAOD.patGenJetTables_cff import configurePatGenJetTables, addPatGenJetDQM
    if not configurePatGenJetTables(process):
        # data: the tables of the generator-level jets do not exist
        for label in ('patGenJetAK4Table', 'patGenJetAK8Table', 'patSubGenJetAK8Table'):
            delattr(process, label)
        del process.patGenJetTablesTask
        print("customisePatGenJetsNano: no MC jet tables in the process (data): nothing added")
        return process
    if hasattr(process, 'nanoTableTaskFS'):
        process.nanoTableTaskFS.add(process.patGenJetTablesTask)
    else:
        raise RuntimeError("customisePatGenJetsNano: the process has no nanoTableTaskFS (not a NanoAOD MC configuration)")
    addPatGenJetDQM(process)
    return process


_OLD_REF = 'genJetFwdRef().backRef()'


def _rewrite_expression(expr):
    """genJetFwdRef().backRef() -> matchedGenJetPtr(): the pat::GenJet link if set, else the original back link (same
    key). A flat replacement: the string parser does not evaluate a ?: nested in another ?: correctly."""
    return expr.replace(_OLD_REF, 'matchedGenJetPtr()')


def _rewrite_parameters(pset, path, changed):
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if isinstance(value, cms.string) and _OLD_REF in value.value():
            value.setValue(_rewrite_expression(value.value()))
            changed.append(path + '.' + name)
        elif isinstance(value, cms.PSet):
            _rewrite_parameters(value, path + '.' + name, changed)
        elif isinstance(value, cms.VPSet):
            for i, p in enumerate(value):
                _rewrite_parameters(p, '%s.%s[%d]' % (path, name, i), changed)
    return changed


def customisePatGenJetsNanoReplace(process, ak8=True):
    from PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD import customiseRestoreSlimmedGenJets
    process = customiseRestoreSlimmedGenJets(process, ak8=ak8)
    changed = []
    for label, module in list(process.producers_().items()) + list(process.filters_().items()) + list(process.analyzers_().items()):
        _rewrite_parameters(module, label, changed)
    print("customisePatGenJetsNanoReplace: %d expressions now follow pat::Jet::matchedGenJetPtr(): %s" % (len(changed), ', '.join(changed)))
    return process
