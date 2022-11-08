import copy
import os
import math
import re
from datetime import date
import Alignment.OfflineValidation.TkAlAllInOneTool.findAndChange as fnc

# Find number of files on a file list. If the list defines a run number before each file, find the number of unique runs instead and return a list of runs with the number
def findNumberOfUnits(fileList):

    with open(fileList,"r") as inputFiles:

        fileContent = inputFiles.readlines()
        firstLine =  fileContent[0].rstrip()
        runsInFiles = []

        # If each line only contains one file, return the number of files
        if len(firstLine.split()) == 1:
            nInputFiles = sum(1 for line in fileContent if line.rstrip())
            return runsInFiles, nInputFiles

        # We now know that the input file is in format "run file". Return the number of unique runs together with the list
        for line in fileContent:
            run = line.split()[0]
            if not run in runsInFiles:
                runsInFiles.append(run)

        return runsInFiles, len(runsInFiles)

def JetHT(config, validationDir):

    # List with all and merge jobs
    jobs = []
    mergeJobs = []
    runType = "single"

    # Find today
    today = date.today()
    dayFormat = today.strftime("%Y-%m-%d")

    # Start with single JetHT jobs
    if not runType in config["validations"]["JetHT"]: 
        raise Exception("No 'single' key word in config for JetHT") 

    for datasetName in config["validations"]["JetHT"][runType]:

        for alignment in config["validations"]["JetHT"][runType][datasetName]["alignments"]:
            # Work directory for each alignment
            workDir = "{}/JetHT/{}/{}/{}".format(validationDir, runType, datasetName, alignment)

            # Write local config
            local = {}
            local["output"] = "{}/{}/JetHT/{}/{}/{}".format(config["LFS"], config["name"], runType, datasetName, alignment)
            local["alignment"] = copy.deepcopy(config["alignments"][alignment])
            local["validation"] = copy.deepcopy(config["validations"]["JetHT"][runType][datasetName])
            local["validation"].pop("alignments")

            useCMSdataset = False
            nInputFiles = 1
            runsInFiles = []
            if "dataset" in config["validations"]["JetHT"][runType][datasetName]:
                inputList = config["validations"]["JetHT"][runType][datasetName]["dataset"]

                # Check if the input is a CMS dataset instead of filelist
                if re.match( r'^/[^/.]+/[^/.]+/[^/.]+$', inputList ):
                    useCMSdataset = True

                # If it is not, read the number of files in a given filelist
                else:
                    runsInFiles, nInputFiles = findNumberOfUnits(inputList)
            else:
                inputList = "needToHaveSomeDefaultFileHere.txt"

            if "filesPerJob" in config["validations"]["JetHT"][runType][datasetName]:
                filesPerJob = config["validations"]["JetHT"][runType][datasetName]["filesPerJob"]
            else:
                filesPerJob = 5

            # If we have defined which runs can be found from which files, we want to define one condor job for run number. Otherwise we do file based splitting.
            oneJobForEachRun = (len(runsInFiles) > 0)
            if oneJobForEachRun:
                nCondorJobs = nInputFiles
                local["runsInFiles"] = runsInFiles
            else:
                nCondorJobs = math.ceil(nInputFiles / filesPerJob)
 
            # Define lines that need to be changed from the template crab configuration
            crabCustomConfiguration = {"overwrite":[], "remove":[], "add":[]}
            crabCustomConfiguration["overwrite"].append("inputList = \'{}\'".format(inputList))
            crabCustomConfiguration["overwrite"].append("jobTag = \'TkAlJetHTAnalysis_{}_{}_{}_{}\'".format(runType, datasetName, alignment, dayFormat))
            crabCustomConfiguration["overwrite"].append("config.Data.unitsPerJob = {}".format(filesPerJob))

            # If there is a CMS dataset defined instead of input file list, make corresponding changes in the configuration file
            if useCMSdataset:
                crabCustomConfiguration["remove"].append("inputList")
                crabCustomConfiguration["remove"].append("config.Data.userInputFiles")
                crabCustomConfiguration["remove"].append("config.Data.totalUnits")
                crabCustomConfiguration["remove"].append("config.Data.outputPrimaryDataset")
                crabCustomConfiguration["overwrite"].pop(0) # Remove inputList from overwrite actions, it is removed for CMS dataset
                crabCustomConfiguration["add"].append("config.Data.inputDataset = \'{}\'".format(inputList))
                crabCustomConfiguration["add"].append("config.Data.inputDBS = \'global\'")
                
            local["crabCustomConfiguration"] = crabCustomConfiguration

            # Write job info
            job = {
                "name": "JetHT_{}_{}_{}".format(runType, alignment, datasetName),
                "dir": workDir,
                "exe": "cmsRun",
                "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/JetHT_cfg.py".format(os.environ["CMSSW_BASE"]),
                "run-mode": "Condor",
                "nCondorJobs": nCondorJobs,
                "exeArguments": "validation_cfg.py config=validation.json jobNumber=$JOBNUMBER",
                "dependencies": [],
                "config": local, 
            }

            jobs.append(job)

    # Merge jobs for JetHT
    if "merge" in config["validations"]["JetHT"]:
        ##List with merge jobs, will be expanded to jobs after looping
        runType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for datasetName in config["validations"]["JetHT"][runType]:

            for alignment in config["validations"]["JetHT"][runType][datasetName]["alignments"]:

                #Work directory for each alignment
                workDir = "{}/JetHT/{}/{}/{}".format(validationDir, runType, datasetName, alignment)

                inputDirectory = "{}/{}/JetHT/single/{}/{}".format(config["LFS"], config["name"], datasetName, alignment)
                outputDirectory = "{}/{}/JetHT/{}/{}/{}".format(config["LFS"], config["name"], runType, datasetName, alignment)

                # Configuration for validateAlignments script
                local = {}
                local["output"] = outputDirectory

                # For eos directory, remove /eos/cms from the beginning of LFS
                eosInputDirectory = inputDirectory
                eosOutputDirectory = outputDirectory

                if inputDirectory.startswith("/eos/cms"):
                    eosInputDirectory = inputDirectory[8:]
                    eosOutputDirectory = outputDirectory[8:]

                # If the directory name starts with /store, we must be working with eos files
                localRun = "true"
                if eosInputDirectory.startswith("/store"):
                    localRun = "false"

                #Write job info
                job = {
                    "name": "JetHT_{}_{}_{}".format(runType, alignment, datasetName),
                    "dir": workDir,
                    "exe": "addHistograms.sh",
                    "exeArguments": "{} {} {} JetHTAnalysis_merged".format(localRun, eosInputDirectory, eosOutputDirectory),
                    "run-mode": "Condor",
                    "flavour": "espresso",
                    "config": local,
                    "dependencies": [],
                }

                ##Loop over all single jobs and set them dependencies for the merge job
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    singleAlignment, singleDatasetName = singleJob["name"].split("_")[2:]

                    if singleDatasetName in config["validations"]["JetHT"][runType][datasetName]["singles"]:
                        if singleAlignment == alignment:
                            job["dependencies"].append(singleJob["name"])

                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    # Plotting for JetHT
    if "plot" in config["validations"]["JetHT"]:
        ##List with merge jobs, will be expanded to jobs after looping
        plotJobs = []
        runType = "plot"

        ##Loop over all merge jobs/IOVs which are wished
        for datasetName in config["validations"]["JetHT"][runType]:

            #Work and output directories for each dataset
            workDir = "{}/JetHT/{}/{}".format(validationDir, runType, datasetName)
            outputDirectory = "{}/{}/JetHT/{}/{}".format(config["LFS"], config["name"], runType, datasetName)

            # Configuration for validateAlignments script
            local = {}
            if "jethtplot" in config["validations"]["JetHT"][runType][datasetName]:
                local["jethtplot"] = copy.deepcopy(config["validations"]["JetHT"][runType][datasetName]["jethtplot"])
            local["output"] = outputDirectory

            # If pT binning changed for validation job, need to change it for plotting also
            if "profilePtBorders" in config["validations"]["JetHT"]["single"][datasetName]:
                local["jethtplot"]["widePtBinBorders"] = config["validations"]["JetHT"]["single"][datasetName]["profilePtBorders"]

            local["jethtplot"]["alignments"] = {}

            # Draw all the alignments for each dataset to same plot
            for alignment in config["validations"]["JetHT"][runType][datasetName]["alignments"]:

                inputDirectory = "{}/{}/JetHT/merge/{}/{}".format(config["LFS"], config["name"], datasetName, alignment)

                eosInputFile = inputDirectory + "/JetHTAnalysis_merged.root"

                # If eos file path is given, remove /eos/cms from the beginning of the file name
                if eosInputFile.startswith("/eos/cms"):
                    eosInputFile = eosInputFile[8:]

                # If the file name starts with /store, add the CERN EOS path to the file name
                if eosInputFile.startswith("/store"):
                    eosInputFile = "root://eoscms.cern.ch/" + eosInputFile

                local["jethtplot"]["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                local["jethtplot"]["alignments"][alignment]["inputFile"] = eosInputFile
                local["jethtplot"]["alignments"][alignment]["legendText"] = config["alignments"][alignment]["title"]

            # Check that luminosity per IOV file is defined
            if not "lumiPerIovFile" in local["jethtplot"]:
                local["jethtplot"]["lumiPerIovFile"] = fnc.digest_path("Alignment/OfflineValidation/data/lumiPerRun_Run2.txt")

            #Write job info
            job = {
                "name": "JetHT_{}_{}".format(runType, datasetName),
                "dir": workDir,
                "exe": "jetHtPlotter",
                "run-mode": "Condor",
                "flavour": "espresso",
                "config": local,
                "dependencies": [],
            }

            ##Loop over all merge jobs and set them dependencies for the plot job
            for mergeJob in mergeJobs:
                ##Get merge job info and append to plot job if requirements are fulfilled
                mergeAlignment, mergeDatasetName = mergeJob["name"].split("_")[2:]

                if mergeDatasetName in config["validations"]["JetHT"][runType][datasetName]["merges"]:
                    job["dependencies"].append(mergeJob["name"])

            plotJobs.append(job)

        jobs.extend(plotJobs)
        
    return jobs
