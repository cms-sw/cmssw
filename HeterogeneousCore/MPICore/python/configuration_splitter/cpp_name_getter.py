import os
import copy
import subprocess
import json
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.MPICore.modules import *

class CPPNameGetter:
    def __init__(
        self,
        original_process,
        helper_file_dir=".cppnamedir",
        cfg_name="print_cppnames_cfg.py",
        json_name="print_cppnames.json",
        log_name="print_cppnames_debug.log",
        reuse=False
    ):
        os.makedirs(helper_file_dir, exist_ok=True)

        self.cfg_path = os.path.join(helper_file_dir, cfg_name)
        self.json_path = os.path.join(helper_file_dir, json_name)
        self.log_name = os.path.join(helper_file_dir, log_name)

        self.reuse = reuse

        if reuse:
            if not os.path.exists(self.json_path):
                raise RuntimeError(
                    f"--cpp-names-exist was specified but JSON file not found: {self.json_path}"
                )
            self.process = None
        else:
            self.process = copy.deepcopy(original_process)

    # ---------- public API ----------

    def get_cpp_types_of_module_products(self):
        if self.reuse:
            return self._read_json()

        self._write_print_cppnames_config()
        self._run_cmsrun()

        return self._read_json()

    # ---------- implementation ----------

    def _write_print_cppnames_config(self):

        self.process.PrintNames = PrintCPPNames(
            outputFile=self.json_path
        )

        self.process.PrintNamesPath = cms.EndPath(self.process.PrintNames)

        if not hasattr(self.process, "schedule") or self.process.schedule is None:
            self.process.schedule = cms.Schedule()

        self.process.schedule.append(self.process.PrintNamesPath)

        self.process.maxEvents.input = 0
        self.process.options.numberOfThreads = 1
        self.process.options.numberOfStreams = 1
        self.process.options.numberOfConcurrentLuminosityBlocks = 1

        with open(self.cfg_path, "w") as f:
            f.write(self.process.dumpPython())

    def _run_cmsrun(self):
        with open(self.log_name, "w") as log:
            subprocess.run(
                ["cmsRun", self.cfg_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False, # Fail later with a nicer message
            )

    def _read_json(self):

        if not os.path.exists(self.json_path):
            raise RuntimeError(
                f"JSON file not produced: {self.json_path}\n"
                "Check that PrintCPPNames analyzer executed correctly at {self.log_name}."
            )

        with open(self.json_path) as f:
            data = json.load(f)

        if not data:
            raise RuntimeError(
                f"JSON file '{self.json_path}' is empty.\n"
                f"Check the analyzer output at {self.log_name}."
            )

        products = defaultdict(list)

        for entry in data:
            products[entry["module"]].append(entry)

        return products
