#!/usr/bin/env python3

# Anzar Afaq         June 17, 2008
# Oleksiy Atramentov June 21, 2008
# Charles Plager     Sept  7, 2010
# Volker Adler       Apr  16, 2014
# Raman Khurana      June 18, 2015
# Dinko Ferencek     June 27, 2015
# Christian Winter   Mar  06, 2025
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re

from FWCore.PythonUtilities.LumiList import LumiList
import json
from datetime import datetime
import subprocess
import Utilities.General.cmssw_das_client as das_client

help = """
How to use:

edmPickEvent.py dataset run1:lumi1:event1 run2:lumi2:event2

- or -

edmPickEvent.py dataset listOfEvents.txt

listOfEvents is a text file:
# this line is ignored as a comment
# since '#' is a valid comment character
run1 lumi_section1 event1
run2 lumi_section2 event2

For example:
# run lum   event
46968   2      4
47011 105     23
47011 140  12312

run, lumi_section, and event are integers that you can get from
edm::Event(Auxiliary)

dataset: it just a name of the physics dataset, if you don't know exact name
    you can provide a mask, e.g.: *QCD*RAW

For updated information see Wiki:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookPickEvents
"""

#################
## Subroutines ##
#################


def getFileNames(run, lumi):
    """Return files for given DAS query via dasgoclient"""
    query = f"file dataset={dataset} run={run} lumi={lumi}"
    cmd = ["dasgoclient", "-query", query, "-json"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = []
    err = proc.stderr.read()
    if err:
        print(f"DAS error: {err}")
        print(proc.stdout.read())
        sys.exit(1)
    else:
        dasout = proc.stdout.read()
        try:
            for row in json.loads(dasout):
                for rec in row.get("file", []):
                    fname = rec.get("name", "")
                    if fname:
                        files.append(fname)
        except:
            print(dasout)
            sys.exit(1)
    return files


def fullCPMpath():
    base = os.environ.get("CMSSW_BASE")
    if not base:
        raise RuntimeError("CMSSW Environment not set")
    retval = f"{base}/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py"
    if os.path.exists(retval):
        return retval
    base = os.environ.get("CMSSW_RELEASE_BASE")
    retval = f"{base}/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py"
    if os.path.exists(retval):
        return retval
    raise RuntimeError("Could not find copyPickMerge_cfg.py")


# crab template
crabTemplate = """
## Edited By Raman Khurana
##
## CRAB documentation : https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCrab
##
## CRAB 3 parameters : https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3ConfigurationFile#CRAB_configuration_parameters
##
## Once you are happy with this file, please run
## crab submit

## In CRAB3 the configuration file is in Python language. It consists of creating a Configuration object imported from the WMCore library: 

from WMCore.Configuration import Configuration
config = Configuration()

##  Once the Configuration object is created, it is possible to add new sections into it with corresponding parameters
config.section_("General")
config.General.requestName = 'pickEvents'
config.General.workArea = 'crab_pickevents_{WorkArea}'


config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '{copyPickMerge}'
config.JobType.pyCfgParams = ['eventsToProcess_load={runEvent}', 'outputFile={output}']

config.section_("Data")
config.Data.inputDataset = '{dataset}'

config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 5
config.Data.lumiMask = '{json}'
#config.Data.publication = True
#config.Data.publishDbsUrl = 'phys03'
#config.Data.publishDataName = 'CRAB3_CSA_DYJets'
#config.JobType.allowNonProductionCMSSW=True

config.section_("Site")
## Change site name accordingly
config.Site.storageSite = "T2_US_Wisconsin"

"""

########################
## ################## ##
## ## Main Program ## ##
## ################## ##
########################

if __name__ == "__main__":

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="""This program
facilitates picking specific events from a data set.  For full details, please visit
https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookPickEvents""",
    )
    parser.add_argument(
        "--output",
        dest="base",
        type=str,
        default="pickevents",
        help='Base name to use for output files (root, JSON, run and event list, etc.)")',
    )
    parser.add_argument(
        "--runInteractive",
        dest="runInteractive",
        action="store_true",
        help='Call "cmsRun" command if possible.  Can take a long time.',
    )
    parser.add_argument(
        "--printInteractive",
        dest="printInteractive",
        action="store_true",
        help='Print "cmsRun" command instead of running it.',
    )
    parser.add_argument(
        "--maxEventsInteractive",
        dest="maxEventsInteractive",
        type=int,
        default=20,
        help="Maximum number of events allowed to be processed interactively.",
    )
    parser.add_argument(
        "--crab",
        dest="crab",
        action="store_true",
        help="Force CRAB setup instead of interactive mode",
    )
    parser.add_argument(
        "--crabCondor",
        dest="crabCondor",
        action="store_true",
        help="Tell CRAB to use Condor scheduler (FNAL or OSG sites).",
    )
    parser.add_argument(
        "--email", dest="email", type=str, default=None, help="Specify email for CRAB"
    )
    parser.add_argument("dataset", type=str, help="Name of the dataset to pick the events from. E.g. '/Muon/Run2022G-22Sep2023-v1/MINIAOD'.")
    parser.add_argument("events", metavar="events", type=str, nargs="+", help="List of 'run:lumi:event' combinations separated by a space or path to a file containing one 'run:lumi:event' combination per line.")
    options = parser.parse_args()

    global dataset  # make dataset a global variable to, so other functions can access it
    dataset = options.dataset

    event_list = (
        set()
    )  # List with all unique events in the form of (run, lumi, event) tuples
    run_lumi_list = (
        set()
    )  # List containing all unique (run, lumi) tuples; this can be considerably smaller than event_list

    if len(options.events) > 1 or ":" in options.events[0]:
        # events are coming in from the command line
        for piece in options.events:
            try:
                run, lumi, event = piece.split(":")
            except:
                raise RuntimeError(f"'{piece}' is not a proper event")
            run_lumi_list.add(
                (int(run), int(lumi))
            )  # only save run and lumi in a tuple, as event is not needed for DAS query
            event_list.add((int(run), int(lumi), int(event)))
    else:
        # read events from file
        with open(options.events[0], "r") as f:
            commentRE = re.compile(r"#.+$")
            for line in f:
                line = commentRE.sub("", line)
                try:
                    run, lumi, event = line.split(":")
                except:
                    print(f"Skipping '{line.strip()}'.")
                    continue
                run_lumi_list.add(
                    (int(run), int(lumi))
                )  # only save run and lumi in a tuple, as event is not needed for DAS query
                event_list.add((int(run), int(lumi), int(event)))

    if not run_lumi_list:
        print("No events defined.  Aborting.")
        sys.exit()

    if len(run_lumi_list) > options.maxEventsInteractive:
        options.crab = True

    if options.crab:

        ##########
        ## CRAB ##
        ##########
        if options.runInteractive:
            raise RuntimeError(
                "This job cannot be run interactively, but rather by crab.  Please call without the '--runInteractive' flag or increase the '--maxEventsInteractive' value."
            )
        run_lumi_list_helper = LumiList(
            lumis=run_lumi_list
        )  # use LumiList as helper to write JSON file
        eventsToProcess = "\n".join(
            sorted(
                [
                    "{run}:{event}".format(run=event_tuple[0], event=event_tuple[2])
                    for event_tuple in event_list
                ]
            )
        )

        # setup the CRAB dictionary
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = options.base
        crabDict = {
            "runEvent": f"{base}_runEvents.txt",
            "copyPickMerge": fullCPMpath(),
            "output": f"{base}.root",
            "crabcfg": f"{base}_crab.py",
            "json": f"{base}.json",
            "dataset": dataset,
            "email": (
                options.email
                if options.email  # guess email from environment if not provided
                else f"{subprocess.getoutput('whoami')}@{'.'.join(subprocess.getoutput('hostname').split('.')[-2:])}"
            ),
            "WorkArea": date,
            "useServer": "",
            "scheduler": "condor" if options.crabCondor else "remoteGlidein",
        }

        run_lumi_list_helper.writeJSON(crabDict["json"])
        with open(crabDict["runEvent"], "w") as f:
            f.write(eventsToProcess + "\n")
        with open(crabDict["crabcfg"], "w") as f:
            f.write(crabTemplate.format(**crabDict))

        print(
            "Please visit CRAB twiki for instructions on how to setup environment for CRAB:\nhttps://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideCrab\n"
        )
        if options.crabCondor:
            print(
                "You are running on condor.  Please make sure you have read instructions on\nhttps://twiki.cern.ch/twiki/bin/view/CMS/CRABonLPCCAF\n"
            )
            if not os.path.exists(f"{os.environ.get('HOME')}/.profile"):
                print(
                    "** WARNING: ** You are missing ~/.profile file.  Please see CRABonLPCCAF instructions above.\n"
                )
        print(
            f"Setup your environment for CRAB and edit {crabDict['crabcfg']} to make any desired changed.  Then run:\n\ncrab submit -c {crabDict['crabcfg']}\n"
        )

    else:

        #################
        ## Interactive ##
        #################
        files = set()
        events_not_in_dataset = []
        # for search of files onl the run and lumisection is relevant. So remove the event and remove the dublicates

        for run, lumi in run_lumi_list:
            print(f"Getting files for run = {run}; lumi = {lumi}", end=": ")
            # Query DAS for files containing the run and lumi
            eventFiles = getFileNames(run, lumi)
            if eventFiles == ["[]"]:  # event not contained in the input dataset
                print(
                    f"\n** WARNING: ** According to a DAS query, run = {run}; lumi = {lumi}; not contained in {dataset}.  Skipping."
                )
                events_not_in_dataset.append((run, lumi))
            else:
                print(f"Found {len(eventFiles)} files")
                files.update(eventFiles)

        # Remove events from the event_list for which no files were found in the dataset
        for event_tuple in event_list:
            if event_tuple[:2] in events_not_in_dataset:
                print("Purging run = {}; lumi = {}; event = {}".format(*event_tuple))
                event_list.remove(event_tuple)

        source = ",".join(files) + "\n"
        eventsToProcess = ",".join(
            sorted([f"{run}:{lumi}:{event}" for run, lumi, event in event_list])
        )
        command = f"edmCopyPickMerge outputFile={options.base}.root \\\n  eventsToProcess={eventsToProcess} \\\n  inputFiles={source}"
        print(
            f"\nYou can now execute the command (also found in 'pickEvents.sh')\n\n{command}"
        )
        with open("pickEvents.sh", "w") as f:
            f.write(f"#!/bin/bash\n{command}")
        if options.runInteractive and not options.printInteractive:
            os.system(command)
