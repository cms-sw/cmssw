#!/usr/bin/env python

from __future__ import print_function

from builtins import range
import os
import re
import sys
import glob
import json
import math
import bisect
import random
import signal
if sys.version_info[0]>2:
  import _pickle as cPickle
else:
  import cPickle
import difflib
import argparse
import functools
import itertools
import subprocess
import collections
import multiprocessing
import FWCore.PythonUtilities.LumiList as LumiList
import Utilities.General.cmssw_das_client as cmssw_das_client
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools


################################################################################
def main(argv = None):
    """
    Main routine. Not called, if this module is loaded via `import`.

    Arguments:
    - `argv`: Command line arguments passed to the script.
    """

    if argv == None:
        argv = sys.argv[1:]

    file_list_creator = FileListCreator(argv)
    file_list_creator.create()


################################################################################
class FileListCreator(object):
    """Create file lists for alignment and validation for a given dataset.
    """

    def __init__(self, argv):
        """Constructor taking the command line arguments.

        Arguments:
        - `args`: command line arguments
        """

        self._first_dataset_ini = True
        self._parser = self._define_parser()
        self._args = self._parser.parse_args(argv)

        if not mps_tools.check_proxy():
            print_msg(
                "Please create proxy via 'voms-proxy-init -voms cms -rfc'.")
            sys.exit(1)

        self._dataset_regex = re.compile(r"^/([^/]+)/([^/]+)/([^/]+)$")
        self._validate_input()

        if self._args.test_mode:
            import Configuration.PyReleaseValidation.relval_steps as rvs
            import Configuration.PyReleaseValidation.relval_production as rvp
            self._args.datasets = [rvs.steps[rvp.workflows[1000][1][0]]["INPUT"].dataSet]
            self._validate_input() # ensure that this change is valid

        self._datasets = sorted([dataset
                                 for pattern in self._args.datasets
                                 for dataset in get_datasets(pattern)
                                 if re.search(self._args.dataset_filter, dataset)])
        if len(self._datasets) == 0:
            print_msg("Found no dataset matching the pattern(s):")
            for d in self._args.datasets: print_msg("\t"+d)
            sys.exit(1)

        self._formatted_dataset = merge_strings(
            [re.sub(self._dataset_regex, r"\1_\2_\3", dataset)
             for dataset in self._datasets])
        self._output_dir = os.path.join(self._args.output_dir,
                                        self._formatted_dataset)
        self._output_dir = os.path.abspath(self._output_dir)
        self._cache = _DasCache(self._output_dir)
        self._prepare_iov_datastructures()
        self._prepare_run_datastructures()

        try:
            os.makedirs(self._output_dir)
        except OSError as e:
            if e.args == (17, "File exists"):
                if self._args.force:
                    pass        # do nothing, just clear the existing output
                elif self._args.use_cache:
                    self._cache.load() # load cache before clearing the output
                else:
                    print_msg("Directory '{}' already exists from previous runs"
                              " of the script. Use '--use-cache' if you want to"
                              " use the cached DAS-query results Or use "
                              "'--force' to remove it."
                              .format(self._output_dir))
                    sys.exit(1)
                files = glob.glob(os.path.join(self._output_dir, "*"))
                for f in files: os.remove(f)
            else:
                raise


    def create(self):
        """Creates file list. To be called by user of the class."""

        self._request_dataset_information()
        self._create_file_lists()
        self._print_eventcounts()
        self._write_file_lists()


    _event_count_log = "event_count_info.log"


    def _define_parser(self):
        """Definition of command line argument parser."""

        parser = argparse.ArgumentParser(
            description = "Create file lists for alignment",
            epilog = ("The tool will create a directory containing all file "
                      "lists and a log file with all relevant event counts "
                      "('{}').".format(FileListCreator._event_count_log)))
        parser.add_argument("-i", "--input", dest = "datasets", required = True,
                            metavar = "DATASET", action = "append",
                            help = ("CMS dataset name; supports wildcards; "
                                    "use multiple times for multiple datasets"))
        parser.add_argument("--dataset-filter", default = "",
                            help = "regex to match within in the datasets matched,"
                                   "in case the wildcard isn't flexible enough")
        parser.add_argument("-j", "--json", dest = "json", metavar = "PATH",
                            help = "path to JSON file (optional)")
        parser.add_argument("-f", "--fraction", dest = "fraction",
                            type = float, default = 1,
                            help = "max. fraction of files used for alignment")
        parser.add_argument("--iov", dest = "iovs", metavar = "RUN", type = int,
                            action = "append", default = [],
                            help = ("define IOV by specifying first run; for "
                                    "multiple IOVs use this option multiple "
                                    "times; files from runs before the lowest "
                                    "IOV are discarded (default: 1)"))
        parser.add_argument("--miniiov", dest="miniiovs", metavar="RUN", type=int,
                            action="append", default=[],
                            help=("in addition to the standard IOVs, break up hippy jobs "
                                  "at these points, so that jobs from before and after "
                                  "these runs are not in the same job"))
        parser.add_argument("-r", "--random", action = "store_true",
                            default = False, help = "select files randomly")
        parser.add_argument("-n", "--events-for-alignment", "--maxevents",
                            dest = "events", type = int, metavar = "NUMBER",
                            help = ("number of events needed for alignment; the"
                                    " remaining events in the dataset are used "
                                    "for validation; if n<=0, all events are "
                                    "used for validation"))
        parser.add_argument("--all-events", action = "store_true",
                            help = "Use all events for alignment")
        parser.add_argument("--tracks-for-alignment", dest = "tracks",
                            type = int, metavar = "NUMBER",
                            help = "number of tracks needed for alignment")
        parser.add_argument("--track-rate", dest = "rate", type = float,
                            metavar = "NUMBER",
                            help = "number of tracks per event")
        parser.add_argument("--run-by-run", dest = "run_by_run",
                            action = "store_true", default = False,
                            help = "create validation file list for each run")
        parser.add_argument("--minimum-events-in-iov",
                            dest = "minimum_events_in_iov", metavar = "NUMBER",
                            type = int, default = 100000,
                            help = ("minimum number of events for alignment per"
                                    " IOV; this option has a higher priority "
                                    "than '-f/--fraction' "
                                    "(default: %(default)s)"))
        parser.add_argument("--minimum-events-validation",
                            dest = "minimum_events_validation",
                            metavar = "NUMBER", type = int, default = 1,
                            help = ("minimum number of events for validation; "
                                    "applies to IOVs; in case of --run-by-run "
                                    "it applies to runs runs "
                                    "(default: %(default)s)"))
        parser.add_argument("--use-cache", dest = "use_cache",
                            action = "store_true", default = False,
                            help = "use DAS-query results of previous run")
        parser.add_argument("-o", "--output-dir", dest = "output_dir",
                            metavar = "PATH", default = os.getcwd(),
                            help = "output base directory (default: %(default)s)")
        parser.add_argument("--create-ini", dest = "create_ini",
                            action = "store_true", default = False,
                            help = ("create dataset ini file based on the "
                                    "created file lists"))
        parser.add_argument("--force", action = "store_true", default = False,
                            help = ("remove output directory from previous "
                                    "runs, if existing"))
        parser.add_argument("--hippy-events-per-job", type = int, default = 1,
                            help = ("approximate number of events in each job for HipPy"))
        parser.add_argument("--test-mode", dest = "test_mode",
                            action = "store_true", default = False,
                            help = argparse.SUPPRESS) # hidden option
        return parser


    def _validate_input(self):
        """Validate command line arguments."""

        if self._args.events is None:
            if self._args.all_events:
                self._args.events = float("inf")
                print_msg("Using all tracks for alignment")
            elif (self._args.tracks is None) and (self._args.rate is None):
                msg = ("either -n/--events-for-alignment, --all-events, or both of "
                       "--tracks-for-alignment and --track-rate are required")
                self._parser.error(msg)
            elif (((self._args.tracks is not None) and (self._args.rate is None)) or
                ((self._args.rate is not None)and (self._args.tracks is None))):
                msg = ("--tracks-for-alignment and --track-rate must be used "
                       "together")
                self._parser.error(msg)
            else:
                self._args.events = int(math.ceil(self._args.tracks /
                                                  self._args.rate))
                print_msg("Requested {0:d} tracks with {1:.2f} tracks/event "
                          "-> {2:d} events for alignment."
                          .format(self._args.tracks, self._args.rate,
                                  self._args.events))
        else:
            if (self._args.tracks is not None) or (self._args.rate is not None) or self._args.all_events:
                msg = ("-n/--events-for-alignment must not be used with "
                       "--tracks-for-alignment, --track-rate, or --all-events")
                self._parser.error(msg)
            print_msg("Requested {0:d} events for alignment."
                      .format(self._args.events))

        for dataset in self._args.datasets:
            if not re.match(self._dataset_regex, dataset):
                print_msg("Dataset pattern '"+dataset+"' is not in CMS format.")
                sys.exit(1)

        nonzero_events_per_iov = (self._args.minimum_events_in_iov > 0)
        if nonzero_events_per_iov and self._args.fraction <= 0:
            print_msg("Setting minimum number of events per IOV for alignment "
                      "to 0 because a non-positive fraction of alignment events"
                      " is chosen: {}".format(self._args.fraction))
            nonzero_events_per_iov = False
            self._args.minimum_events_in_iov = 0
        if nonzero_events_per_iov and self._args.events <= 0:
            print_msg("Setting minimum number of events per IOV for alignment "
                      "to 0 because a non-positive number of alignment events"
                      " is chosen: {}".format(self._args.events))
            nonzero_events_per_iov = False
            self._args.minimum_events_in_iov = 0


    def _prepare_iov_datastructures(self):
        """Create the needed objects for IOV handling."""

        self._iovs = sorted(set(self._args.iovs))
        if len(self._iovs) == 0: self._iovs.append(1)
        self._iov_info_alignment = {iov: {"events": 0, "files": []}
                                        for iov in self._iovs}
        self._iov_info_validation = {iov: {"events": 0, "files": []}
                                         for iov in self._iovs}

        self._miniiovs = sorted(set(self._iovs) | set(self._args.miniiovs))


    def _get_iovs(self, runs, useminiiovs=False):
        """
        Return the IOV start for `run`. Returns 'None' if the run is before any
        defined IOV.

        Arguments:
        - `runs`: run numbers
        """

        iovlist = self._miniiovs if useminiiovs else self._iovs

        iovs = []
        for run in runs:
          iov_index = bisect.bisect(iovlist, run)
          if iov_index > 0: iovs.append(iovlist[iov_index-1])
        return iovs


    def _prepare_run_datastructures(self):
        """Create the needed objects for run-by-run validation file lists."""

        self._run_info = {}


    def _add_file_info(self, container, keys, fileinfo):
        """Add file with `file_name` to `container` using `key`.

        Arguments:
        - `container`: dictionary holding information on files and event counts
        - `keys`: keys to which the info should be added; will be created if not
                  existing
        - `file_name`: name of a dataset file
        """

        for key in keys:
            if key not in container:
                container[key] = {"events": 0,
                                  "files": []}
            container[key]["events"] += fileinfo.nevents / len(keys)
            if fileinfo not in container[key]["files"]:
                container[key]["files"].append(fileinfo)


    def _remove_file_info(self, container, keys, fileinfo):
        """Remove file with `file_name` to `container` using `key`.

        Arguments:
        - `container`: dictionary holding information on files and event counts
        - `keys`: keys from which the info should be removed
        - `file_name`: name of a dataset file
        - `event_count`: number of events in `file_name`
        """

        for key in keys:
            if key not in container: continue
            try:
                index = container[key]["files"].index(fileinfo)
            except ValueError:      # file not found
                return
            del container[key]["files"][index]
            container[key]["events"] -= fileinfo.nevents / len(keys)


    def _request_dataset_information(self):
        """Retrieve general dataset information and create file list."""

        if not self._cache.empty:
            print_msg("Using cached information.")
            (self._events_in_dataset,
             self._files,
             self._file_info,
             self._max_run) = self._cache.get()
            self.rereco = any(len(fileinfo.runs)>1 for fileinfo in self._file_info)
            if self._args.random: random.shuffle(self._files)
            return

        # workaround to deal with KeyboardInterrupts in the worker processes:
        # - ignore interrupt signals in workers (see initializer)
        # - use a timeout of size sys.maxsize to avoid a bug in multiprocessing
        number_of_processes = multiprocessing.cpu_count() - 1
        number_of_processes = (number_of_processes
                               if number_of_processes > 0
                               else 1)
        pool = multiprocessing.Pool(
            processes = number_of_processes,
            initializer = lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

        print_msg("Requesting information for the following dataset(s):")
        for d in self._datasets: print_msg("\t"+d)
        print_msg("This may take a while...")

        result = pool.map_async(get_events_per_dataset, self._datasets).get(3600)
        self._events_in_dataset = sum(result)

        result = pool.map_async(get_max_run, self._datasets).get(3600)
        self._max_run = max(result)

        result = sum(pool.map_async(get_file_info, self._datasets).get(3600), [])
        files = pool.map_async(_make_file_info, result).get(3600)
        self._file_info = sorted(fileinfo for fileinfo in files)

        self.rereco = any(len(fileinfo.runs)>1 for fileinfo in self._file_info)

        if self._args.test_mode:
            self._file_info = self._file_info[-200:] # take only last chunk of files
        self._files = [fileinfo.name for fileinfo in self._file_info]

        # write information to cache
        self._cache.set(self._events_in_dataset, self._files, self._file_info,
                        self._max_run)
        self._cache.dump()
        if self._args.random:
          random.shuffle(self._file_info)
          self._files = [fileinfo.name for fileinfo in self._file_info]

    def _create_file_lists(self):
        """Create file lists for alignment and validation."""

        # collect files for alignment until minimal requirements are fulfilled
        self._files_alignment = []
        self._files_validation = []
        self._events_for_alignment = 0
        self._events_for_validation = 0

        max_range = (0
                     if self._args.events <= 0
                     else int(math.ceil(len(self._files)*self._args.fraction)))
        use_for_alignment = True
        for i, fileinfo in enumerate(self._file_info):
            enough_events = self._events_for_alignment >= self._args.events
            fraction_exceeded = i >= max_range
            if enough_events or fraction_exceeded: use_for_alignment = False

            dataset, f, number_of_events, runs = fileinfo

            iovs = self._get_iovs(runs)
            if use_for_alignment:
                if iovs:
                    self._events_for_alignment += number_of_events
                    self._files_alignment.append(fileinfo)
                    self._add_file_info(self._iov_info_alignment, iovs, fileinfo)
                else:
                    max_range += 1 # not used -> discard in fraction calculation
            else:
                if iovs:
                    self._events_for_validation += number_of_events
                    self._files_validation.append(fileinfo)
                    self._add_file_info(self._iov_info_validation, iovs, fileinfo)
                    if self._args.run_by_run:
                        self._add_file_info(self._run_info, runs, fileinfo)

        self._fulfill_iov_eventcount()

        self._split_hippy_jobs()


    def _fulfill_iov_eventcount(self):
        """
        Try to fulfill the requirement on the minimum number of events per IOV
        in the alignment file list by picking files from the validation list.
        """

        for iov in self._iovs:
            if self._iov_info_alignment[iov]["events"] >= self._args.minimum_events_in_iov: continue
            for fileinfo in self._files_validation[:]:
                dataset, f, number_of_events, runs = fileinfo
                iovs = self._get_iovs(runs)
                if iov in iovs:
                    self._files_alignment.append(fileinfo)
                    self._events_for_alignment += number_of_events
                    self._add_file_info(self._iov_info_alignment, iovs, fileinfo)

                    self._events_for_validation -= number_of_events
                    self._remove_file_info(self._iov_info_validation, iovs, fileinfo)
                    if self._args.run_by_run:
                        self._remove_file_info(self._run_info, runs, fileinfo)
                    self._files_validation.remove(fileinfo)

                    if (self._iov_info_alignment[iov]["events"]
                        >= self._args.minimum_events_in_iov):
                        break   # break the file loop if already enough events

    def _split_hippy_jobs(self):
        hippyjobs = {}
        for dataset, miniiov in itertools.product(self._datasets, self._miniiovs):
            jobsforminiiov = []
            hippyjobs[dataset,miniiov] = jobsforminiiov
            eventsinthisjob = float("inf")
            for fileinfo in self._files_alignment:
                if fileinfo.dataset != dataset: continue
                miniiovs = set(self._get_iovs(fileinfo.runs, useminiiovs=True))
                if miniiov not in miniiovs: continue
                if len(miniiovs) > 1:
                    hippyjobs[dataset,miniiov] = []
                if eventsinthisjob >= self._args.hippy_events_per_job:
                    currentjob = []
                    jobsforminiiov.append(currentjob)
                    eventsinthisjob = 0
                currentjob.append(fileinfo)
                currentjob.sort()
                eventsinthisjob += fileinfo.nevents

        self._hippy_jobs = {
          (dataset, iov): sum((hippyjobs[dataset, miniiov]
                               for miniiov in self._miniiovs
                               if iov == max(_ for _ in self._iovs if _ <= miniiov)), []
                           )
          for dataset, iov in itertools.product(self._datasets, self._iovs)
        }

    def _print_eventcounts(self):
        """Print the event counts per file list and per IOV."""

        log = os.path.join(self._output_dir, FileListCreator._event_count_log)

        print_msg("Using {0:d} events for alignment ({1:.2f}%)."
                  .format(self._events_for_alignment,
                          100.0*
                          self._events_for_alignment/self._events_in_dataset),
                  log_file = log)
        for iov in sorted(self._iov_info_alignment):
            print_msg(("Approximate events" if self.rereco else "Events") + " for alignment in IOV since {0:f}: {1:f}"
                      .format(iov, self._iov_info_alignment[iov]["events"]),
                      log_file = log)

        print_msg("Using {0:d} events for validation ({1:.2f}%)."
                  .format(self._events_for_validation,
                          100.0*
                          self._events_for_validation/self._events_in_dataset),
                  log_file = log)

        for iov in sorted(self._iov_info_validation):
            msg = ("Approximate events" if self.rereco else "Events") + " for validation in IOV since {0:f}: {1:f}".format(
                iov, self._iov_info_validation[iov]["events"])
            if (self._iov_info_validation[iov]["events"]
                < self._args.minimum_events_validation):
                msg += " (not enough events -> no dataset file will be created)"
            print_msg(msg, log_file = log)

        for run in sorted(self._run_info):
            msg = ("Approximate events" if self.rereco else "Events") + " for validation in run {0:f}: {1:f}".format(
                run, self._run_info[run]["events"])
            if (self._run_info[run]["events"]
                < self._args.minimum_events_validation):
                msg += " (not enough events -> no dataset file will be created)"
            print_msg(msg, log_file = log)

        unused_events = (self._events_in_dataset
                         - self._events_for_validation
                         - self._events_for_alignment)
        if unused_events > 0 != self._events_in_dataset:
            print_msg("Unused events: {0:d} ({1:.2f}%)"
                      .format(unused_events,
                              100.0*unused_events/self._events_in_dataset),
                      log_file = log)


    def _create_dataset_ini_section(self, name, collection, json_file = None):
        """Write dataset ini snippet.

        Arguments:
        - `name`: name of the dataset section
        - `collection`: track collection of this dataset
        - `json_file`: JSON file to be used for this dataset (optional)
        """

        if json_file:
            splitted = name.split("_since")
            file_list = "_since".join(splitted[:-1]
                                      if len(splitted) > 1
                                      else splitted)
        else:
            file_list = name
        output = "[dataset:{}]\n".format(name)
        output += "collection = {}\n".format(collection)
        output += "inputFileList = ${{datasetdir}}/{}.txt\n".format(file_list)
        output += "json = ${{datasetdir}}/{}\n".format(json_file) if json_file else ""

        if collection in ("ALCARECOTkAlCosmicsCTF0T",
                          "ALCARECOTkAlCosmicsInCollisions"):
            if self._first_dataset_ini:
                print_msg("\tDetermined cosmics dataset, i.e. please replace "
                          "'DUMMY_DECO_MODE_FLAG' and 'DUMMY_ZERO_TESLA_FLAG' "
                          "with the correct values.")
                self._first_dataset_ini = False
            output += "cosmicsDecoMode  = DUMMY_DECO_MODE_FLAG\n"
            output += "cosmicsZeroTesla = DUMMY_ZERO_TESLA_FLAG\n"
        output += "\n"

        return output


    def _create_json_file(self, name, first, last = None):
        """
        Create JSON file with `name` covering runs from `first` to `last`.  If a
        global JSON is provided, the resulting file is the intersection of the
        file created here and the global one.
        Returns the name of the created JSON file.

        Arguments:
        - `name`: name of the creted JSON file
        - `first`: first run covered by the JSON file
        - `last`: last run covered by the JSON file

        """

        if last is None: last = self._max_run
        name += "_JSON.txt"
        print_msg("Creating JSON file: "+name)

        json_file = LumiList.LumiList(runs = range(first, last+1))
        if self._args.json:
            global_json = LumiList.LumiList(filename = self._args.json)
            json_file = json_file & global_json
        json_file.writeJSON(os.path.join(self._output_dir, name))

        return name


    def _get_track_collection(self, edm_file):
        """Extract track collection from given `edm_file`.

        Arguments:
        - `edm_file`: CMSSW dataset file
        """

        # use global redirector to allow also files not yet at your site:
        cmd = ["edmDumpEventContent", r"root://cms-xrd-global.cern.ch/"+edm_file]
        try:
            event_content = subprocess.check_output(cmd).split("\n")
        except subprocess.CalledProcessError as e:
            splitted = edm_file.split("/")
            try:
                alcareco = splitted[splitted.index("ALCARECO")+1].split("-")[0]
                alcareco = alcareco.replace("TkAlCosmics0T", "TkAlCosmicsCTF0T")
                alcareco = "ALCARECO" + alcareco
                print_msg("\tDetermined track collection as '{}'.".format(alcareco))
                return alcareco
            except ValueError:
                if "RECO" in splitted:
                    print_msg("\tDetermined track collection as 'generalTracks'.")
                    return "generalTracks"
                else:
                    print_msg("\tCould not determine track collection "
                              "automatically.")
                    print_msg("\tPlease replace 'DUMMY_TRACK_COLLECTION' with "
                              "the correct value.")
                    return "DUMMY_TRACK_COLLECTION"

        track_collections = []
        for line in event_content:
            splitted = line.split()
            if len(splitted) > 0 and splitted[0] == r"vector<reco::Track>":
                track_collections.append(splitted[1].strip().strip('"'))
        if len(track_collections) == 0:
            print_msg("No track collection found in file '{}'.".format(edm_file))
            sys.exit(1)
        elif len(track_collections) == 1:
            print_msg("\tDetermined track collection as "
                      "'{}'.".format(track_collections[0]))
            return track_collections[0]
        else:
            alcareco_tracks = filter(lambda x: x.startswith("ALCARECO"),
                                     track_collections)
            if len(alcareco_tracks) == 0 and "generalTracks" in track_collections:
                print_msg("\tDetermined track collection as 'generalTracks'.")
                return "generalTracks"
            elif len(alcareco_tracks) == 1:
                print_msg("\tDetermined track collection as "
                          "'{}'.".format(alcareco_tracks[0]))
                return alcareco_tracks[0]
            print_msg("\tCould not unambiguously determine track collection in "
                      "file '{}':".format(edm_file))
            print_msg("\tPlease replace 'DUMMY_TRACK_COLLECTION' with "
                      "the correct value from the following list.")
            for collection in track_collections:
                print_msg("\t - "+collection)
            return "DUMMY_TRACK_COLLECTION"


    def _write_file_lists(self):
        """Write file lists to disk."""

        self._create_dataset_txt(self._formatted_dataset, self._files_alignment)
        self._create_hippy_txt(self._formatted_dataset, sum(self._hippy_jobs.values(), []))
        self._create_dataset_cff(
            "_".join(["Alignment", self._formatted_dataset]),
            self._files_alignment)

        self._create_dataset_cff(
            "_".join(["Validation", self._formatted_dataset]),
            self._files_validation)


        if self._args.create_ini:
            dataset_ini_general = "[general]\n"
            dataset_ini_general += "datasetdir = {}\n".format(self._output_dir)
            dataset_ini_general += ("json = {}\n\n".format(self._args.json)
                                    if self._args.json
                                    else "\n")

            ini_path = self._formatted_dataset + ".ini"
            print_msg("Creating dataset ini file: " + ini_path)
            ini_path = os.path.join(self._output_dir, ini_path)

            collection = self._get_track_collection(self._files[0])

            with open(ini_path, "w") as f:
                f.write(dataset_ini_general)
                f.write(self._create_dataset_ini_section(
                    self._formatted_dataset, collection))

            iov_wise_ini = dataset_ini_general

        for i,iov in enumerate(sorted(self._iovs)):
            iov_str = "since{0:d}".format(iov)
            iov_str = "_".join([self._formatted_dataset, iov_str])

            if self.rereco:
                if i == len(self._iovs) - 1:
                    last = None
                else:
                    last = sorted(self._iovs)[i+1] - 1
                local_json = self._create_json_file(iov_str, iov, last)
            else:
                local_json = None

            if self._args.create_ini:
                iov_wise_ini += self._create_dataset_ini_section(iov_str,
                                                                 collection,
                                                                 local_json)

            self._create_dataset_txt(iov_str,
                                     self._iov_info_alignment[iov]["files"])
            self._create_hippy_txt(iov_str, sum((self._hippy_jobs[dataset,iov] for dataset in self._datasets), []))
            self._create_dataset_cff(
                "_".join(["Alignment", iov_str]),
                self._iov_info_alignment[iov]["files"],
                json_file=local_json)

            if (self._iov_info_validation[iov]["events"]
                < self._args.minimum_events_validation):
                continue
            self._create_dataset_cff(
                "_".join(["Validation", iov_str]),
                self._iov_info_validation[iov]["files"],
                json_file=local_json)

        if self._args.create_ini and iov_wise_ini != dataset_ini_general:
            ini_path = self._formatted_dataset + "_IOVs.ini"
            print_msg("Creating dataset ini file: " + ini_path)
            ini_path = os.path.join(self._output_dir, ini_path)
            with open(ini_path, "w") as f: f.write(iov_wise_ini)

        for run in sorted(self._run_info):
            if args.rereco: continue #need to implement more jsons
            if (self._run_info[run]["events"]
                < self._args.minimum_events_validation):
                continue
            self._create_dataset_cff(
                "_".join(["Validation", self._formatted_dataset, str(run)]),
                self._run_info[run]["files"])


    def _create_dataset_txt(self, name, file_list):
        """Write alignment file list to disk.

        Arguments:
        - `name`: name of the file list
        - `file_list`: list of files to write to `name`
        """

        name += ".txt"
        print_msg("Creating dataset file list: "+name)
        with open(os.path.join(self._output_dir, name), "w") as f:
            f.write("\n".join(fileinfo.name for fileinfo in file_list))


    def _create_hippy_txt(self, name, job_list):
        name += "_hippy.txt"
        print_msg("Creating dataset file list for HipPy: "+name)
        with open(os.path.join(self._output_dir, name), "w") as f:
            f.write("\n".join(",".join("'"+fileinfo.name+"'" for fileinfo in job) for job in job_list)+"\n")


    def _create_dataset_cff(self, name, file_list, json_file = None):
        """
        Create configuration fragment to define a dataset.

        Arguments:
        - `name`: name of the configuration fragment
        - `file_list`: list of files to write to `name`
        - `json_file`: JSON file to be used for this dataset (optional)
        """

        if json_file is None: json_file = self._args.json # might still be None
        if json_file is not None:
            json_file = os.path.join(self._output_dir, json_file)

        name = "_".join(["Dataset",name, "cff.py"])
        print_msg("Creating dataset configuration fragment: "+name)

        file_list_str = ""
        for sub_list in get_chunks(file_list, 255):
            file_list_str += ("readFiles.extend([\n'"+
                              "',\n'".join(fileinfo.name for fileinfo in sub_list)+
                              "'\n])\n")

        fragment = FileListCreator._dataset_template.format(
            lumi_def = ("import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                        "lumiSecs = cms.untracked.VLuminosityBlockRange()\n"
                        "goodLumiSecs = LumiList.LumiList(filename = "
                        "'{0:s}').getCMSSWString().split(',')"
                        .format(json_file)
                        if json_file else ""),
            lumi_arg = ("lumisToProcess = lumiSecs,\n                    "
                        if json_file else ""),
            lumi_extend = "lumiSecs.extend(goodLumiSecs)" if json_file else "",
            files = file_list_str)

        with open(os.path.join(self._output_dir, name), "w") as f:
            f.write(fragment)


    _dataset_template = """\
import FWCore.ParameterSet.Config as cms
{lumi_def:s}
readFiles = cms.untracked.vstring()
source = cms.Source("PoolSource",
                    {lumi_arg:s}fileNames = readFiles)
{files:s}{lumi_extend:s}
maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
"""


class _DasCache(object):
    """Helper class to cache information from DAS requests."""

    def __init__(self, file_list_id):
        """Constructor of the cache.

        Arguments:
        - `file_list_id`: ID of the cached file lists
        """

        self._file_list_id = file_list_id
        self._cache_file_name = os.path.join(file_list_id, ".das_cache.pkl")
        self.reset()


    def reset(self):
        """Reset the cache contents and the 'empty' flag."""

        self._empty = True
        self._events_in_dataset = 0
        self._files = []
        self._file_info = []
        self._max_run = None


    def set(self, total_events, file_list, file_info, max_run):
        """Set the content of the cache.

        Arguments:
        - `total_events`: total number of events in dataset
        - `file_list`: list of files in dataset
        - `file_info`: dictionary with numbers of events per file
        - `max_run`: highest run number contained in the dataset
        """

        self._events_in_dataset = total_events
        self._files = file_list
        self._file_info = file_info
        self._max_run = max_run
        self._empty = False


    def get(self):
        """
        Get the content of the cache as tuple:
           result = (total number of events in dataset,
                     list of files in dataset,
                     dictionary with numbers of events and runs per file)
        """

        return self._events_in_dataset, self._files, self._file_info, self._max_run


    def load(self):
        """Loads the cached contents."""

        if not self.empty:
            print_msg("Overriding file information with cached information.")
        try:
            with open(self._cache_file_name, "rb") as f:
                tmp_dict = cPickle.load(f)
                self.__dict__.update(tmp_dict)
        except IOError as e:
            if e.args == (2, "No such file or directory"):
                msg = "Failed to load cache for '{}'.".format(self._file_list_id)
                if not self.empty:
                    msg += " Keeping the previous file information."
                print_msg(msg)
            else:
                raise


    def dump(self):
        """Dumps the contents to the cache file."""

        if self.empty:
            print_msg("Cache is empty. Not writing to file.")
            return

        with open(self._cache_file_name, "wb") as f:
            cPickle.dump(self.__dict__, f, 2)


    @property
    def empty(self):
        """
        Flag indicating whether the cache is empty or has been filled (possibly
        with nothing).
        """

        return self._empty



################################################################################
def das_client(query, check_key = None):
    """
    Submit `query` to DAS client and handle possible errors.
    Further treatment of the output might be necessary.

    Arguments:
    - `query`: DAS query
    - `check_key`: optional key to be checked for; retriggers query if needed
    """

    error = True
    for i in range(5):         # maximum of 5 tries
        try:
            das_data = cmssw_das_client.get_data(query, limit = 0)
        except IOError as e:
            if e.errno == 14: #https://stackoverflow.com/q/36397853/5228524
                continue
        except ValueError as e:
            if str(e) == "No JSON object could be decoded":
                continue

        if das_data["status"] == "ok":
            if das_data["nresults"] == 0 or check_key is None:
                error = False
                break

            result_count = 0
            for d in find_key(das_data["data"], [check_key]):
                result_count += len(d)
            if result_count == 0:
                das_data["status"] = "error"
                das_data["reason"] = ("DAS did not return required data.")
                continue
            else:
                error = False
                break

    if das_data["status"] == "error":
        print_msg("DAS query '{}' failed 5 times. "
                  "The last time for the the following reason:".format(query))
        print(das_data["reason"])
        sys.exit(1)
    return das_data["data"]


def find_key(collection, key_chain):
    """Searches for `key` in `collection` and returns first corresponding value.

    Arguments:
    - `collection`: list of dictionaries
    - `key_chain`: chain of keys to be searched for
    """

    result = None
    for i,key in enumerate(key_chain):
        for item in collection:
            if key in item:
                if i == len(key_chain) - 1:
                    result = item[key]
                else:
                    try:
                        result = find_key(item[key], key_chain[i+1:])
                    except LookupError:
                        pass    # continue with next `item` in `collection`
            else:
                pass            # continue with next `item` in `collection`

    if result is not None: return result
    raise LookupError(key_chain, collection) # put


def print_msg(text, line_break = True, log_file = None):
    """Formatted printing of `text`.

    Arguments:
    - `text`: string to be printed
    """

    msg = "  >>> " + str(text)
    if line_break:
        print(msg)
    else:
        print(msg, end=' ')
        sys.stdout.flush()
    if log_file:
        with open(log_file, "a") as f: f.write(msg+"\n")
    return msg


def get_runs(file_name):
    """
    Try to guess the run number from `file_name`. If run could not be
    determined, gets the run numbers from DAS (slow!)

    Arguments:
    - `file_name`: name of the considered file
    """
    try:
        return [int("".join(file_name.split("/")[-4:-2]))]
    except ValueError:
        query = "run file="+file_name+" system=dbs3"
        return [int(_) for _ in find_key(das_client(query), ["run", "run_number"])]


def get_max_run(dataset_name):
    """Retrieve the maximum run number in `dataset_name`.

    Arguments:
    - `dataset_name`: name of the dataset
    """

    data = das_client("run dataset={0:s} system=dbs3".format(dataset_name))
    runs = [f["run"][0]["run_number"] for f in data]
    return max(runs)


def get_files(dataset_name):
    """Retrieve list of files in `dataset_name`.

    Arguments:
    - `dataset_name`: name of the dataset
    """

    data = das_client(("file dataset={0:s} system=dbs3 detail=True | "+
                       "grep file.name, file.nevents > 0").format(dataset_name),
                      "file")
    return [find_key(f["file"], ["name"]) for f in data]


def get_datasets(dataset_pattern):
    """Retrieve list of dataset matching `dataset_pattern`.

    Arguments:
    - `dataset_pattern`: pattern of dataset names
    """

    data = das_client("dataset dataset={0:s} system=dbs3 detail=True"
                      "| grep dataset.name".format(dataset_pattern), "dataset")
    return sorted(set([find_key(f["dataset"], ["name"]) for f in data]))


def get_events_per_dataset(dataset_name):
    """Retrieve the number of a events in `dataset_name`.

    Arguments:
    - `dataset_name`: name of a dataset
    """

    return _get_events("dataset", dataset_name)


def get_events_per_file(file_name):
    """Retrieve the number of a events in `file_name`.

    Arguments:
    - `file_name`: name of a dataset file
    """

    return _get_events("file", file_name)


def _get_events(entity, name):
    """Retrieve the number of events from `entity` called `name`.

    Arguments:
    - `entity`: type of entity
    - `name`: name of entity
    """

    data = das_client("{0:s}={1:s} system=dbs3 detail=True | grep {0:s}.nevents"
                      .format(entity, name), entity)
    return int(find_key(data, [entity, "nevents"]))


def _get_properties(name, entity, properties, filters = None, sub_entity = None,
                    aggregators = None):
    """Retrieve `properties` from `entity` called `name`.

    Arguments:
    - `name`: name of entity
    - `entity`: type of entity
    - `properties`: list of property names
    - `filters`: list of filters on properties
    - `sub_entity`: type of entity from which to extract the properties;
                    defaults to `entity`
    - `aggregators`: additional aggregators/filters to amend to query
    """

    if sub_entity is None: sub_entity = entity
    if filters is None:    filters    = []
    props = ["{0:s}.{1:s}".format(sub_entity,prop.split()[0])
             for prop in properties]
    conditions = ["{0:s}.{1:s}".format(sub_entity, filt)
                  for filt in filters]
    add_ons = "" if aggregators is None else " | "+" | ".join(aggregators)

    data = das_client("{0:s} {1:s}={2:s} system=dbs3 detail=True | grep {3:s}{4:s}"
                      .format(sub_entity, entity, name,
                              ", ".join(props+conditions), add_ons), sub_entity)
    return [[find_key(f[sub_entity], [prop]) for prop in properties] for f in data]

def get_file_info(dataset):
    result = _get_properties(name=dataset,
                             properties = ["name", "nevents"],
                             filters = ["nevents > 0"],
                             entity = "dataset",
                             sub_entity = "file")
    return [(dataset, name, nevents) for name, nevents in result]



FileInfo = collections.namedtuple("FileInfo", "dataset name nevents runs")

def _make_file_info(dataset_name_nevents):
    return FileInfo(*dataset_name_nevents, runs=get_runs(dataset_name_nevents[1]))

def get_chunks(long_list, chunk_size):
    """
    Generates list of sub-lists of `long_list` with a maximum size of
    `chunk_size`.

    Arguments:
    - `long_list`: original list
    - `chunk_size`: maximum size of created sub-lists
    """

    for i in range(0, len(long_list), chunk_size):
        yield long_list[i:i+chunk_size]


def merge_strings(strings):
    """Merge strings in `strings` into a common string.

    Arguments:
    - `strings`: list of strings
    """

    if type(strings) == str:
        return strings
    elif len(strings) == 0:
        return ""
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        first = strings[0]
        second = strings[1]
    else:
        first = merge_strings(strings[:-1])
        second = strings[-1]

    merged_string = ""
    blocks = difflib.SequenceMatcher(None, first, second).get_matching_blocks()

    last_i, last_j, last_n = 0, 0, 0
    for i, j, n in blocks:
        merged_string += first[last_i+last_n:i]
        merged_string += second[last_j+last_n:j]
        merged_string += first[i:i+n]
        last_i, last_j, last_n = i, j, n

    return str(merged_string)


################################################################################
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
