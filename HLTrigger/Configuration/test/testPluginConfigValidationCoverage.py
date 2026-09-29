#!/usr/bin/env python3

import os
import re
import sys
import tempfile
import subprocess
import importlib.util
import inspect
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

KNOWN_OFFENDERS = {
    "DTRecHitProducer",
    "DTRecSegment4DProducer",
    "DetIdAssociatorESProducer",
    "EmptyESSource",
    "GsfMaterialEffectsESProducer",
    "HGCalRecHitProducer",
    "MuonIdProducer",
    "TrajectoryCleanerESProducer",
}

def build_loaded_module_map():
    scram_arch = os.environ["SCRAM_ARCH"]

    base_vars = (
        "CMSSW_RELEASE_BASE",
        "CMSSW_FULL_RELEASE_BASE",
        "CMSSW_BASE",
    )

    search_dirs = [
        os.path.join(os.environ[var], "cfipython", scram_arch)
        for var in base_vars
        if var in os.environ
    ]

    patterns = [
        re.compile(r"cms\.EDProducer\(['\"]([^'\"]+)['\"]"),
        re.compile(r"cms\.EDFilter\(['\"]([^'\"]+)['\"]"),
        re.compile(r"cms\.EDAnalyzer\(['\"]([^'\"]+)['\"]"),
        re.compile(r"cms\.ESProducer\(['\"]([^'\"]+)['\"]"),
        re.compile(r"cms\.ESSource\(['\"]([^'\"]+)['\"]"),
    ]

    plugin_types = set()

    for base in search_dirs:
        if not os.path.exists(base):
            continue

        for root, _, files in os.walk(base):
            for fname in files:
                if not fname.endswith("_cfi.py"):
                    continue

                path = os.path.join(root, fname)

                with open(path) as f:
                    content = f.read()

                match = re.search(r"from \.(\w+) import", content)

                if match:
                    impl = os.path.join(root, match.group(1) + ".py")
                    if os.path.exists(impl):
                        with open(impl) as f:
                            content = f.read()

                for pattern in patterns:
                    plugin_types.update(pattern.findall(content))

    return plugin_types


def dump_hlt_menu():
    with tempfile.TemporaryDirectory() as tmpdir:

        cfg = os.path.join(tmpdir, "cfg.py")
        dump = os.path.join(tmpdir, "dump.py")

        with open(cfg, "w") as f:
            f.write(
                """
import FWCore.ParameterSet.Config as cms
process = cms.Process("HLT")
process.load("HLTrigger.Configuration.HLT_75e33_cff")
"""
            )

        with open(dump, "w") as f:
            subprocess.check_call(
                ["edmConfigDump", cfg],
                stdout=f,
            )

        spec = importlib.util.spec_from_file_location("dump", dump)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod.process, mod.cms


def extract_plugin_types(process, cms_module):
    plugin_types = set()

    for _, obj in inspect.getmembers(process):
        if isinstance(
            obj,
            (
                cms_module.EDProducer,
                cms_module.EDFilter,
                cms_module.EDAnalyzer,
                cms_module.ESProducer,
                cms_module.ESSource,
            ),
        ):
            plugin_type = (
                repr(obj)
                .split("(")[1]
                .split(" ")[0]
                .strip()
                .replace(",", "")
                .replace('"', "")
                .replace(")", "")
            )

            if plugin_type:
                plugin_types.add(plugin_type)

    return plugin_types


def main():
    available_plugins = build_loaded_module_map()

    process, cms_module = dump_hlt_menu()
    menu_plugins = extract_plugin_types(process, cms_module)

    missing = sorted(
        (menu_plugins - available_plugins) - KNOWN_OFFENDERS
    )

    ignored = sorted(
        (menu_plugins - available_plugins) & KNOWN_OFFENDERS
    )

    if ignored:
        print("Ignoring known offenders:")
        for p in ignored:
            print(f"  {p}")

    if missing:
        print("ERROR: plugin types not found in loaded modules:")
        for p in missing:
            print(f"  {p}")
        return 1

    print(
        f"SUCCESS: checked {len(menu_plugins)} plugin types, "
        f"all found (excluding {len(ignored)} whitelisted offenders)."
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
