#!/usr/bin/env python3
"""Generate and validate the standalone Phase-2 L1 configuration (L1_Phase2_cff + L1_Phase2/).

The L1 content is the difference between two cmsDriver configurations of the reference
workflow: one running only the base step (DIGI) and one running base + L1 steps. Only
objects reachable from the L1 paths are kept; the Run-3 legacy chain (everything feeding
only simGtStage2Digis) is moved to its own task and path.

  l1tPhase2Fragment.py generate [--outdir L1Trigger/Configuration/python]
  l1tPhase2Fragment.py validate

See L1Trigger/Configuration/python/L1_Phase2/README.md and https://github.com/artlbv/cmssw/issues/10
"""
import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

import FWCore.ParameterSet.Config as cms

# reference workflow: 29634.0 (TTbar_14TeV, Run4D110), step2
DRIVER_OPTIONS = [
    "--conditions", "auto:phase2_realistic_T35",
    "--geometry", "ExtendedRun4D110",
    "--era", "Phase2C17I13M9",
    "--datatier", "GEN-SIM-DIGI-RAW",
    "--eventcontent", "FEVTDEBUGHLT",
    "--filein", "file:step1.root",
    "--fileout", "file:step2.root",
    "-n", "10",
    "--mc",
    "--no_exec",
]
BASE_STEPS = "DIGI:pdigi_valid"
L1_STEPS = "L1TrackTrigger,L1,L1P2GT"

# modules whose ancestors (unless also needed by Phase-2 objects) form the Run-3 legacy chain
LEGACY_ROOTS = ["simGtStage2Digis", "simBmtfDigis", "simCscTriggerPrimitiveDigisRun3"]
LEGACY_TASK = "L1TLegacyRun3Task"
LEGACY_PATH = "L1simulationLegacy_step"
L1_PATH_AFTER_WHICH_LEGACY = "L1simulation_step"

FRAGMENT = "L1_Phase2"
PACKAGE = "L1Trigger/Configuration"

MODULE_TYPES = (cms.EDProducer, cms.EDFilter, cms.EDAnalyzer)
SEQUENCE_TYPES = (cms.Sequence, cms.Task)


def runDriver(steps, cfgFile):
    cmd = ["cmsDriver.py", "step2", "-s", steps, "--python_filename", cfgFile] + DRIVER_OPTIONS
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def loadProcess(cfgFile):
    spec = importlib.util.spec_from_file_location(os.path.basename(cfgFile)[:-3], cfgFile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process


def driverProcesses(workdir):
    base, full = os.path.join(workdir, "base_cfg.py"), os.path.join(workdir, "full_cfg.py")
    runDriver(BASE_STEPS, base)
    runDriver(BASE_STEPS + "," + L1_STEPS, full)
    return loadProcess(base), loadProcess(full)


class Collector:
    """Collects module and sequence/task labels reachable from the visited paths."""

    def __init__(self):
        self.modules = set()
        self.sequences = set()

    def enter(self, node):
        if isinstance(node, MODULE_TYPES):
            self.modules.add(node.label_())
        elif isinstance(node, SEQUENCE_TYPES) and node.hasLabel_():
            self.sequences.add(node.label_())

    def leave(self, node):
        pass


def reachable(paths):
    collector = Collector()
    for path in paths:
        path.visit(collector)
    return collector


def inputTags(pset):
    for name in pset.parameterNames_():
        par = getattr(pset, name)
        if isinstance(par, cms.InputTag):
            yield par
        elif isinstance(par, cms.VInputTag):
            yield from (t if isinstance(t, cms.InputTag) else cms.InputTag(t) for t in par)
        elif isinstance(par, cms.PSet):
            yield from inputTags(par)
        elif isinstance(par, cms.VPSet):
            for p in par:
                yield from inputTags(p)


def ancestors(roots, deps):
    seen, stack = set(), list(roots)
    while stack:
        label = stack.pop()
        if label not in seen:
            seen.add(label)
            stack.extend(deps.get(label, ()))
    return seen


def l1Content(base, full):
    """Returns the L1 paths, modules, sequences, ES objects, legacy modules and external inputs."""
    basePaths = {p.label_() for p in base.schedule}
    paths = [p for p in full.schedule if p.label_() not in basePaths]
    reached = reachable(paths)
    modules = reached.modules
    deps = {m: {t.getModuleLabel() for t in inputTags(getattr(full, m))} & modules for m in modules}
    legacyChain = ancestors([r for r in LEGACY_ROOTS if r in modules], deps)
    legacy = legacyChain - ancestors(modules - legacyChain, deps)
    externalInputs = sorted({t.getModuleLabel() for m in modules for t in inputTags(getattr(full, m))
                             if t.getModuleLabel() and t.getModuleLabel() not in modules})
    es = {}
    for kind in ("es_sources_", "es_producers_", "es_prefers_"):
        objs, baseObjs = getattr(full, kind)(), getattr(base, kind)()
        for label in objs:
            if label not in baseObjs:
                es[label] = objs[label]
            elif objs[label].dumpPython() != baseObjs[label].dumpPython():
                raise RuntimeError(f"ES object '{label}' is modified by the L1 steps")
    return paths, modules, reached.sequences, es, legacy, externalInputs


def splitLegacy(full, paths, sequences, legacy):
    """Removes the legacy modules from the L1 sequences/tasks and schedules them in their own path."""
    containers = [getattr(full, s) for s in sequences]
    for label in sorted(legacy):
        module = getattr(full, label)
        for container in containers:
            while container.remove(module):
                pass
    setattr(full, LEGACY_TASK, cms.Task(*[getattr(full, m) for m in sorted(legacy)]))
    setattr(full, LEGACY_PATH, cms.Path(getattr(full, LEGACY_TASK)))
    index = [p.label_() for p in paths].index(L1_PATH_AFTER_WHICH_LEGACY) + 1
    paths.insert(index, getattr(full, LEGACY_PATH))
    # drop the containers that became empty
    remaining = reachable(paths).sequences
    empty = {s for s in remaining if not reachable([getattr(full, s)]).modules}
    for label in sorted(empty):
        for container in containers + paths:
            while container.remove(getattr(full, label)):
                pass
    return reachable(paths).sequences


HEADER = "import FWCore.ParameterSet.Config as cms\n\n"


def dumpContainer(obj, kinds):
    """Dump of a path/sequence/task with 'process.' references turned into imports."""
    text = obj.dumpPython()
    names = sorted(set(re.findall(r"process\.(\w+)", text)))
    imports = "".join(f"from ..{kinds[n]}.{n}_cfi import *\n" for n in names)
    return imports, re.sub(r"process\.", "", text)


def writeFile(directory, label, imports, body):
    with open(os.path.join(directory, label + "_cfi.py"), "w") as out:
        out.write(HEADER + imports + ("\n" if imports else "") + f"{label} = {body}")


def generate(args):
    with tempfile.TemporaryDirectory() as workdir:
        base, full = driverProcesses(workdir)
    paths, modules, sequences, es, legacy, externalInputs = l1Content(base, full)
    sequences = splitLegacy(full, paths, sequences, legacy)

    top = os.path.join(args.outdir, FRAGMENT)
    for kind in ("modules", "eventsetup", "sequences", "paths"):
        directory = os.path.join(top, kind)
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    kinds = {m: "modules" for m in modules}
    kinds.update({s: "sequences" for s in sequences})
    kinds.update({p.label_(): "paths" for p in paths})

    for label in sorted(modules):
        writeFile(os.path.join(top, "modules"), label, "", getattr(full, label).dumpPython())
    for label in sorted(es):
        writeFile(os.path.join(top, "eventsetup"), label, "", es[label].dumpPython())
    for label in sorted(sequences):
        writeFile(os.path.join(top, "sequences"), label, *dumpContainer(getattr(full, label), kinds))
    for path in paths:
        writeFile(os.path.join(top, "paths"), path.label_(), *dumpContainer(path, kinds))

    loadPrefix = f"{PACKAGE}/{FRAGMENT}"
    lines = [HEADER.rstrip("\n"), "",
             "# Standalone, fully resolved configuration of the Phase-2 L1 trigger emulation",
             "# (track trigger, trigger primitives, correlator/GTT/GMT/calo emulators, P2GT menu),",
             "# modelled on HLTrigger/Configuration/python/HLT_75e33_cff.py.",
             "#",
             "# GENERATED by L1Trigger/Configuration/test/l1tPhase2Fragment.py: do not edit by hand.",
             f"#   cmsDriver.py step2 -s {BASE_STEPS},{L1_STEPS} {' '.join(DRIVER_OPTIONS)}",
             f"#   release: {os.environ.get('CMSSW_VERSION', 'unknown')}",
             "# Geometry, magnetic field and GlobalTag are not included (taken from the driver).",
             "# Required inputs (not produced here): " + ", ".join(externalInputs),
             "",
             f'fragment = cms.ProcessFragment("L1")',
             "",
             "### Event-setup (ES sources, ES producers, ES prefers)"]
    lines += [f'fragment.load("{loadPrefix}/eventsetup/{label}_cfi")' for label in sorted(es)]
    lines += ["", "### Paths (modules and sequences are imported by the path files)"]
    lines += [f'fragment.load("{loadPrefix}/paths/{p.label_()}_cfi")' for p in paths]
    lines += ["", "fragment.schedule = cms.Schedule(*["]
    lines += [f"    fragment.{p.label_()}," for p in paths]
    lines += ["])", ""]
    with open(os.path.join(args.outdir, FRAGMENT + "_cff.py"), "w") as out:
        out.write("\n".join(lines))

    print(f"wrote {len(modules)} modules ({len(legacy)} Run-3 legacy), {len(es)} ES objects, "
          f"{len(sequences)} sequences/tasks, {len(paths)} paths to {top}")
    print("required inputs:", ", ".join(externalInputs))


def validate(args):
    """Compares base + L1_Phase2_cff with the cmsDriver process: modules, parameters, ES, schedule."""
    with tempfile.TemporaryDirectory() as workdir:
        base, full = driverProcesses(workdir)
        test = loadProcess(os.path.join(workdir, "base_cfg.py"))
    refPaths, refModules, _, refES, legacy, _ = l1Content(base, full)

    test.load(f"{PACKAGE.replace('/', '.')}.{FRAGMENT}_cff")
    testPaths = [getattr(test, p.label_()) for p in refPaths if hasattr(test, p.label_())]
    if hasattr(test, LEGACY_PATH):
        testPaths.append(getattr(test, LEGACY_PATH))
    testModules = reachable(testPaths).modules
    errors = []

    missing, extra = refModules - testModules, testModules - refModules
    if missing:
        errors.append(f"modules missing from the fragment: {sorted(missing)}")
    if extra:
        errors.append(f"modules not in the cmsDriver process: {sorted(extra)}")
    for label in sorted(refModules & testModules):
        if getattr(full, label).dumpPython() != getattr(test, label).dumpPython():
            errors.append(f"parameters differ for module '{label}'")

    testES = {}
    for kind in ("es_sources_", "es_producers_", "es_prefers_"):
        testES.update(getattr(test, kind)())
    for label, obj in refES.items():
        if label not in testES:
            errors.append(f"ES object missing from the fragment: '{label}'")
        elif obj.dumpPython() != testES[label].dumpPython():
            errors.append(f"parameters differ for ES object '{label}'")

    refOrder = [p.label_() for p in refPaths]
    testOrder = [p for p in refOrder if hasattr(test, p)]
    if testOrder != refOrder:
        errors.append(f"paths missing from the fragment: {sorted(set(refOrder) - set(testOrder))}")

    for e in errors:
        print("ERROR:", e)
    print(f"checked {len(refModules)} modules, {len(refES)} ES objects, {len(refOrder)} paths: "
          + ("FAILED" if errors else "OK"))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="regenerate L1_Phase2_cff.py and L1_Phase2/")
    gen.add_argument("--outdir", default=os.path.join(os.environ.get("CMSSW_BASE", "."), "src", PACKAGE, "python"))
    sub.add_parser("validate", help="compare the fragment with the cmsDriver configuration")
    args = parser.parse_args()
    return generate(args) if args.command == "generate" else validate(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
