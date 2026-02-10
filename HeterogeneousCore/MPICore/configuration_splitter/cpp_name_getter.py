import os
import copy
import subprocess
from collections import defaultdict

import FWCore.ParameterSet.Config as cms


class CPPNameGetter:
    def __init__(
        self,
        original_process,
        helper_file_dir=".cppnamedir",
        cfg_name="print_cppnames_cfg.py",
        log_name="print_cppnames.log",
        exists=False
    ):
        """
        Run PrintCPPNames on a copy of `process` and return product C++ types.

        Returns:
          dict[module] -> list of dicts:
            instance, product_instance, process, type, branch
        """

        os.makedirs(helper_file_dir, exist_ok=True)

        self.cfg_path = os.path.join(helper_file_dir, cfg_name)
        self.log_path = os.path.join(helper_file_dir, log_name)

        if exists:
            if not os.path.exists(self.log_path):
                raise RuntimeError(
                    f"--cpp-names-exist was specified, but the C++ names file was not found: {self.log_path}"
                )
            self.process = None
        else:
            self.process = copy.deepcopy(original_process)
        self.exists = exists
    

    # ---------- public API ----------

    def get_cpp_types_of_module_products(self):
        if self.exists and os.path.exists(self.log_path):
            # reuse existing output
            return self._parse_cpp_types()

        # otherwise regenerate
        self._write_print_cppnames_config()
        self._run_cmsrun()

        return self._parse_cpp_types()


    # ---------- implementation details ----------

    def _write_print_cppnames_config(self):

        # add printer
        self.process.PrintNames = cms.EDProducer("PrintCPPNames")
        self.process.PrintNamesPath = cms.Path(self.process.PrintNames)

        if not hasattr(self.process, "schedule") or self.process.schedule is None:
            self.process.schedule = cms.Schedule()

        self.process.schedule.append(self.process.PrintNamesPath)

        # force zero-event job
        self.process.maxEvents.input = 0
        self.process.options.numberOfThreads = 1
        self.process.options.numberOfStreams = 1
        self.process.options.numberOfConcurrentLuminosityBlocks = 1

        with open(self.cfg_path, "w") as f:
            f.write(self.process.dumpPython())

    def _run_cmsrun(self):
        with open(self.log_path, "w") as log:
            subprocess.run(
                ["cmsRun", self.cfg_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,  # framework exits are expected
            )

    def _parse_cpp_types(self):
        """
        Returns:
          dict[module] -> list of dicts:
            instance, product_instance, process, type, branch
        """
        products = defaultdict(list)

        with open(self.log_path) as f:
            for line in f:
                if "PrintCPPNames: considering product " not in line:
                    continue

                try:
                    lhs, rhs = line.split(" of type ", 1)
                    type_part, branch_part = rhs.split(" branch type ", 1)
                except ValueError:
                    continue

                branch = branch_part.strip()

                full = lhs.split("considering product ", 1)[1].strip()

                # instance = everything before first "_"
                if "_" not in full:
                    continue
                instance, rest = full.split("_", 1)

                # rest = <module>_<productInstance>_<process>
                parts = rest.rsplit("_", 2)
                if len(parts) != 3:
                    continue

                module, product_instance, process = parts

                products[module].append({
                    "instance": instance,
                    "product_instance": product_instance,
                    "process": process,
                    "type": type_part.strip(),
                    "branch": branch,
                })

        return products
