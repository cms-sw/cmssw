#!/usr/bin/env python

import os
import re
import sys
import glob
import json
import math
import bisect
import random
import signal
import cPickle
import difflib
import argparse
import subprocess
import multiprocessing
import Utilities.General.cmssw_das_client as cmssw_das_client


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

        if not check_proxy():
            print_msg(
                "Please create proxy via 'voms-proxy-init -voms cms -rfc'.")
            sys.exit(1)

        self._dataset_regex = re.compile(r"^/([^/]+)/([^/]+)/([^/]+)$")
        parser = self._define_parser()
        self._args = parser.parse_args(argv)
        self._validate_input()
        self._datasets = sorted([dataset
                                 for pattern in self._args.datasets
                                 for dataset in get_datasets(pattern)])

        self._formatted_dataset = merge_strings(
            [re.sub(self._dataset_regex, r"\1_\2_\3", dataset)
             for dataset in self._datasets])
        self._cache = _DasCache(self._formatted_dataset)
        self._prepare_iov_datastructures()
        self._prepare_run_datastructures()

        try:
            os.makedirs(self._formatted_dataset)
        except OSError as e:
            if e.args == (17, "File exists"):
                if self._args.use_cache:
                    self._cache.load()
                    files = glob.glob(os.path.join(self._formatted_dataset, "*"))
                    for f in files: os.remove(f)
                else:
                    print_msg("Directory '{}' already exists from previous runs"
                              " of the script. Use '--use-cache' if you want to"
                              " use the cached DAS-query results. Otherwise, "
                              "remove it, please."
                              .format(self._formatted_dataset))
                    sys.exit(1)
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
        parser.add_argument("-j", "--json", dest = "json", metavar = "PATH",
                            help = "path to JSON file (optional)")
        parser.add_argument("-f", "--fraction", dest = "fraction",
                            type = float, default = 0.5,
                            help = "max. fraction of files used for alignment")
        parser.add_argument("--iov", dest = "iovs", metavar = "RUN", type = int,
                            action = "append", default = [],
                            help = ("define IOV by specifying first run; for "
                                    "multiple IOVs use this option multiple "
                                    "times; files from runs before the lowest "
                                    "IOV are discarded (default: 1)"))
        parser.add_argument("-r", "--random", action = "store_true",
                            default = False, help = "select files randomly")
        parser.add_argument("-n", "--events-for-alignment", dest = "events",
                            type = int, metavar = "NUMBER",
                            help = ("number of events needed for alignment; the"
                                    " remaining events in the dataset are used "
                                    "for validation; if n<=0, all events are "
                                    "used for validation"))
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
        return parser


    def _validate_input(self):
        """Validate command line arguments."""

        if self._args.events:
            if self._args.tracks or self._args.rate:
                msg = ("-n/--events-for-alignment must not be used with "
                       "--tracks-for-alignment or --track-rate")
                parser.error(msg)
            print_msg("Requested {0:d} events for alignment."
                      .format(self._args.events))
        else:
            if not (self._args.tracks and self._args.rate):
                msg = ("--tracks-for-alignment and --track-rate must be used "
                       "together")
                parser.error(msg)
            self._args.events = int(math.ceil(self._args.tracks /
                                              self._args.rate))
            print_msg("Requested {0:d} tracks with {1:.2f} tracks/event "
                      "-> {2:d} events for alignment."
                      .format(self._args.tracks, self._args.rate,
                              self._args.events))

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
        self._iov_info_alignment = dict((iov, {"events": 0, "files": []})
                                        for iov in self._iovs)
        self._iov_info_validation = dict((iov, {"events": 0, "files": []})
                                         for iov in self._iovs)


    def _get_iov(self, run):
        """
        Return the IOV start for `run`. Returns 'None' if the run is before any
        defined IOV.

        Arguments:
        - `run`: run number
        """

        iov_index = bisect.bisect(self._iovs, run)
        if iov_index > 0: return self._iovs[iov_index-1]
        else: return None


    def _prepare_run_datastructures(self):
        """Create the needed objects for run-by-run validation file lists."""

        self._run_info = {}


    def _add_file_info(self, container, key, file_name):
        """Add file with `file_name` to `container` using `key`.

        Arguments:
        - `container`: dictionary holding information on files and event counts
        - `key`: key to which the info should be added; will be created if not
                 existing
        - `file_name`: name of a dataset file
        """

        if key not in container:
            container[key] = {"events": 0,
                              "files": []}
        container[key]["events"] += self._file_info[file_name]
        container[key]["files"].append(file_name)


    def _remove_file_info(self, container, key, file_name):
        """Remove file with `file_name` to `container` using `key`.

        Arguments:
        - `container`: dictionary holding information on files and event counts
        - `key`: key from which the info should be removed
        - `file_name`: name of a dataset file
        - `event_count`: number of events in `file_name`
        """

        if key not in container: return
        try:
            index = container[key]["files"].index(file_name)
        except ValueError:      # file not found
            return
        del container[key]["files"][index]
        container[key]["events"] -= self._file_info[file_name]


    def _request_dataset_information(self):
        """Retrieve general dataset information and create file list."""

        if not self._cache.empty:
            print_msg("Using cached information.")
            (self._events_in_dataset,
             self._files,
             self._file_info) = self._cache.get()
            return

        self._events_in_dataset = 0
        self._files = []
        for dataset in self._datasets:
            print_msg("Requesting information for dataset '{0:s}'."
                      .format(dataset))
            self._events_in_dataset += get_events_per_dataset(dataset)
            self._files.extend(get_files(dataset))
        if self._args.random: random.shuffle(self._files)

        result = print_msg("Counting events in {0:d} dataset files. This may "
                           "take several minutes...".format(len(self._files)))
        # workaround to deal with KeyboardInterrupts in the worker processes:
        # - ignore interrupt signals in workers (see initializer)
        # - use a timeout of size sys.maxint to avoid a bug in multiprocessing
        number_of_processes = multiprocessing.cpu_count() - 1
        number_of_processes = (number_of_processes
                               if number_of_processes > 0
                               else 1)
        pool = multiprocessing.Pool(
            processes = number_of_processes,
            initializer = lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        count = pool.map_async(get_events_per_file, self._files).get(sys.maxsize)
        self._file_info = dict(zip(self._files, count))

        # write information to cache
        self._cache.set(self._events_in_dataset, self._files, self._file_info)
        self._cache.dump()


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
        for i, f in enumerate(self._files):
            enough_events = self._events_for_alignment >= self._args.events
            fraction_exceeded = i >= max_range
            if enough_events or fraction_exceeded: use_for_alignment = False

            number_of_events = self._file_info[f]
            run = guess_run(f)
            iov = self._get_iov(run)
            if use_for_alignment:
                if iov:
                    self._events_for_alignment += number_of_events
                    self._files_alignment.append(f)
                    self._add_file_info(self._iov_info_alignment, iov, f)
                else:
                    max_range += 1 # not used -> discard in fraction calculation
            else:
                if iov:
                    self._events_for_validation += number_of_events
                    self._files_validation.append(f)
                    self._add_file_info(self._iov_info_validation, iov, f)
                    if self._args.run_by_run:
                        self._add_file_info(self._run_info, run, f)

        self._fulfill_iov_eventcount()


    def _fulfill_iov_eventcount(self):
        """
        Try to fulfill the requirement on the minimum number of events per IOV
        in the alignment file list by picking files from the validation list.
        """

        not_enough_events = [
            iov for iov in self._iovs
            if (self._iov_info_alignment[iov]["events"]
                < self._args.minimum_events_in_iov)]
        for iov in not_enough_events:
            for f in self._files_validation:
                run = guess_run(f)
                if self._get_iov(run) == iov:
                    self._files_alignment.append(f)
                    number_of_events = self._file_info[f]
                    self._events_for_alignment += number_of_events
                    self._add_file_info(self._iov_info_alignment, iov, f)

                    self._events_for_validation -= number_of_events
                    self._remove_file_info(self._iov_info_validation, iov, f)
                    if self._args.run_by_run:
                        self._remove_file_info(self._run_info, run, f)

                    if (self._iov_info_alignment[iov]["events"]
                        >= self._args.minimum_events_in_iov):
                        break   # break the file loop if already enough events

        self._files_validation = [f for f in self._files_validation
                                  if f not in self._files_alignment]


    def _print_eventcounts(self):
        """Print the event counts per file list and per IOV."""

        log = os.path.join(self._formatted_dataset,
                           FileListCreator._event_count_log)

        print_msg("Using {0:d} events for alignment ({1:.2f}%)."
                  .format(self._events_for_alignment,
                          100.0*
                          self._events_for_alignment/self._events_in_dataset),
                  log_file = log)
        for iov in sorted(self._iov_info_alignment):
            print_msg("Events for alignment in IOV since {0:d}: {1:d}"
                      .format(iov, self._iov_info_alignment[iov]["events"]),
                      log_file = log)

        print_msg("Using {0:d} events for validation ({1:.2f}%)."
                  .format(self._events_for_validation,
                          100.0*
                          self._events_for_validation/self._events_in_dataset),
                  log_file = log)
        for iov in sorted(self._iov_info_validation):
            msg = "Events for validation in IOV since {0:d}: {1:d}".format(
                iov, self._iov_info_validation[iov]["events"])
            if (self._iov_info_validation[iov]["events"]
                < self._args.minimum_events_validation):
                msg += " (not enough events -> no dataset file will be created)"
            print_msg(msg, log_file = log)

        for run in sorted(self._run_info):
            msg = "Events for validation in run {0:d}: {1:d}".format(
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


    def _write_file_lists(self):
        """Write file lists to disk."""

        self._create_alignment_file_list(self._formatted_dataset+".txt",
                                         self._files_alignment)

        self._create_validation_dataset("_".join(["Dataset",
                                                  self._formatted_dataset,
                                                  "cff.py"]),
                                        self._files_validation)

        for iov in sorted(self._iovs):
            iov_str = "since{0:d}".format(iov)
            self._create_alignment_file_list(
                "_".join([self._formatted_dataset, iov_str])+".txt",
                self._iov_info_alignment[iov]["files"])

            if (self._iov_info_validation[iov]["events"]
                < self._args.minimum_events_validation):
                continue
            self._create_validation_dataset(
                "_".join(["Dataset", self._formatted_dataset, iov_str, "cff.py"]),
                self._iov_info_validation[iov]["files"])

        for run in sorted(self._run_info):
            if (self._run_info[run]["events"]
                < self._args.minimum_events_validation):
                continue
            self._create_validation_dataset(
                "_".join(["Dataset", self._formatted_dataset, str(run), "cff.py"]),
                self._run_info[run]["files"])



    def _create_alignment_file_list(self, name, file_list):
        """Write alignment file list to disk.

        Arguments:
        - `name`: name of the file list
        - `file_list`: list of files to written to `name`
        """

        print_msg("Creating MillePede file list: "+name)
        with open(os.path.join(self._formatted_dataset, name), "w") as f:
            f.write("\n".join(file_list))


    def _create_validation_dataset(self, name, file_list):
        """
        Create configuration fragment to define a dataset for validation.

        Arguments:
        - `name`: name of the configuration fragment
        - `file_list`: list of files to written to `name`
        """

        print_msg("Creating validation dataset configuration fragment: "+name)

        file_list_str = ""
        for sub_list in get_chunks(file_list, 255):
            file_list_str += ("readFiles.extend([\n'"+
                              "',\n'".join(sub_list)+
                              "'\n])\n")

        fragment = FileListCreator._dataset_template.format(
            lumi_def = ("import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                        "lumiSecs = cms.untracked.VLuminosityBlockRange()\n"
                        "goodLumiSecs = LumiList.LumiList(filename = "
                        "'{0:s}').getCMSSWString().split(',')"
                        .format(self._args.json)
                        if self._args.json else ""),
            lumi_arg = ("lumisToProcess = lumiSecs,\n                    "
                        if self._args.json else ""),
            lumi_extend = ("lumiSecs.extend(goodLumiSecs)"
                           if self._args.json else ""),
            files = file_list_str)

        with open(os.path.join(self._formatted_dataset, name), "w") as f:
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


    def set(self, total_events, file_list, file_info):
        """Set the content of the cache.

        Arguments:
        - `total_events`: total number of events in dataset
        - `file_list`: list of files in dataset
        - `file_info`: dictionary with numbers of events per file
        """

        self._events_in_dataset = total_events
        self._files = file_list
        self._file_info = file_info
        self._empty = False


    def get(self):
        """
        Get the content of the cache as tuple:
           result = (total number of events in dataset,
                     list of files in dataset,
                     dictionary with numbers of events per file)
        """

        return self._events_in_dataset, self._files, self._file_info


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
def das_client(query):
    """
    Submit `query` to DAS client and handle possible errors.
    Further treatment of the output might be necessary.

    Arguments:
    - `query`: DAS query
    """
    for _ in xrange(3):         # maximum of 3 tries
        das_data = cmssw_das_client.get_data(query, limit = 0)
        if das_data["status"] != "error": break
    if das_data["status"] == "error":
        print_msg("DAS query '{}' failed 3 times. "
                  "The last time for the the following reason:".format(query))
        print das_data["reason"]
        sys.exit(1)
    return das_data["data"]


def find_key(collection, key):
    """Searches for `key` in `collection` and returns first corresponding value.

    Arguments:
    - `collection`: list of dictionaries
    - `key`: key to be searched for
    """

    for item in collection:
        if key in item:
            return item[key]
    print collection
    raise KeyError(key)


def print_msg(text, line_break = True, log_file = None):
    """Formatted printing of `text`.

    Arguments:
    - `text`: string to be printed
    """

    msg = "  >>> " + str(text)
    if line_break:
        print msg
    else:
        print msg,
        sys.stdout.flush()
    if log_file:
        with open(log_file, "a") as f: f.write(msg+"\n")
    return msg


def guess_run(file_name):
    """
    Try to guess the run number from `file_name`. If run could not be
    determined, 'sys.maxint' is returned.

    Arguments:
    - `file_name`: name of the considered file
    """
    try:
        return int("".join(file_name.split("/")[-4:-2]))
    except ValueError:
        return sys.maxsize


def get_files(dataset_name):
    """Retrieve list of files in `dataset_name`.

    Arguments:
    - `dataset_name`: name of the dataset
    """

    data = das_client("file dataset={0:s} | grep file.name, file.nevents > 0"
                      .format(dataset_name))
    return [find_key(f["file"], "name") for f in data]


def get_datasets(dataset_pattern):
    """Retrieve list of dataset matching `dataset_pattern`.

    Arguments:
    - `dataset_pattern`: pattern of dataset names
    """

    data = das_client("dataset dataset={0:s} | grep dataset.name"
                      .format(dataset_pattern))
    return [find_key(f["dataset"], "name") for f in data]


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

    data = das_client("{0:s}={1:s} | grep {0:s}.nevents".format(entity, name))
    return int(find_key(find_key(data, entity), "nevents"))


def get_chunks(long_list, chunk_size):
    """
    Generates list of sub-lists of `long_list` with a maximum size of
    `chunk_size`.

    Arguments:
    - `long_list`: original list
    - `chunk_size`: maximum size of created sub-lists
    """

    for i in xrange(0, len(long_list), chunk_size):
        yield long_list[i:i+chunk_size]


def check_proxy():
    """Check if GRID proxy has been initialized."""

    try:
        with open(os.devnull, "w") as dump:
            subprocess.check_call(["voms-proxy-info", "--exists"],
                                  stdout = dump, stderr = dump)
    except subprocess.CalledProcessError:
        return False
    return True


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
