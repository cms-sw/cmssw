import copy
import json
import os
import subprocess
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.MPICore.modules import DumpProductNames


class DumpJob:
    """
    A short cmsRun job that asks the framework about the configuration, and
    gets the answer as json. Subclasses add the components that write the json
    files, in configure(), and read the files back, in parse().
    """

    directory = ".edmMpiSplitConfig"
    name = None
    json_names = ()

    def __init__(self, process, reuse=False):
        """
        With `reuse`, read the json files of an earlier run instead of running
        cmsRun again.
        """
        os.makedirs(self.directory, exist_ok=True)
        self.cfg_path = os.path.join(self.directory, f"{self.name}_cfg.py")
        self.log_path = os.path.join(self.directory, f"{self.name}.log")

        self.reuse = reuse
        # deepcopy the original process, so that the original process is not touched
        self.process = None if reuse else copy.deepcopy(process)

        if reuse:
            for json_name in self.json_names:
                if not os.path.exists(self.json_path(json_name)):
                    raise RuntimeError(f"asked to reuse '{self.json_path(json_name)}', but that file does not exist")

    def json_path(self, json_name):
        """The path of one of the json files of the job."""
        return os.path.join(self.directory, f"{json_name}.json")

    def run(self):
        """Prepare and run the job. Return the parsed result."""
        if not self.reuse:
            self.configure(self.process)
            self.process.maxEvents.input = 0
            self.process.options.numberOfThreads = 1
            self.process.options.numberOfStreams = 1
            self.process.options.numberOfConcurrentLuminosityBlocks = 1

            with open(self.cfg_path, "w") as cfg:
                cfg.write(self.process.dumpPython())

            # Delete pre-existing old json files, if any
            for json_name in self.json_names:
                if os.path.exists(self.json_path(json_name)):
                    os.remove(self.json_path(json_name))

            with open(self.log_path, "w") as log:
                # Not check=True: a missing or empty JSON gives a much clearer message.
                subprocess.run(["cmsRun", self.cfg_path], stdout=log, stderr=subprocess.STDOUT, check=False)

        return self.parse()

    def load(self, json_name):
        """Read a json file, return its contents"""
        path = self.json_path(json_name)
        if not os.path.exists(path):
            raise RuntimeError(f"'{path}' was not produced; check the log at {self.log_path}")

        with open(path) as f:
            data = json.load(f)
        if not data:
            raise RuntimeError(f"'{path}' is empty; check the log at {self.log_path}")

        return data


class ProcessDump(DumpJob):
    """
    The job that runs the DumpDependencyGraph service and the DumpProductNames
    analyzer, to get the module dependency graph and the data products of the
    process.
    """

    name = "process_dump"
    json_names = ("product_names", "dependency_graph")

    def configure(self, process):
        """
        Add the `DumpDependencyGraph` service and a `DumpProductNames`
        analyzer on a new EndPath, `PrintNamesPath`.
        """
        process.DumpDependencyGraph = cms.Service(
            "DumpDependencyGraph",
            fileName=cms.untracked.string(self.json_path("dependency_graph")),
        )

        process.PrintNames = DumpProductNames(outputFile=self.json_path("product_names"))
        process.PrintNamesPath = cms.EndPath(process.PrintNames)

        if getattr(process, "schedule", None) is not None:
            process.schedule.append(process.PrintNamesPath)

    def parse(self):
        """
        Return the pair (products, graph). `products` groups the entries of
        `product_names.json` by module label. `graph` is the content of
        `dependency_graph.json`.
        """
        products = defaultdict(list)
        for entry in self.load("product_names"):
            products[entry["module"]].append(entry)

        return products, self.load("dependency_graph")
