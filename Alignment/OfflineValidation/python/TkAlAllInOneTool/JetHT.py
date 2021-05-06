import copy
import os

def JetHT(config, validationDir):
    ##List with all and merge jobs
    jobs = []
    mergeJobs = []
    runType = "single"

    ##Start with single DMR jobs
    if not runType in config["validations"]["JetHT"]: 
        raise Exception("No 'single' key word in config for JetHT") 

    for datasetName in config["validations"]["JetHT"][runType]:

        for alignment in config["validations"]["JetHT"][runType][datasetName]["alignments"]:
            ##Work directory for each alignment
            workDir = "{}/JetHT/{}/{}/{}".format(validationDir, runType, datasetName, alignment)

            ##Write local config
            local = {}
            local["output"] = "{}/{}/JetHT/{}/{}/{}".format(config["LFS"], config["name"], runType, datasetName, alignment)
            local["alignment"] = copy.deepcopy(config["alignments"][alignment])
            local["validation"] = copy.deepcopy(config["validations"]["JetHT"][runType][datasetName])
            local["validation"].pop("alignments")

            ##Write job info
            job = {
                "name": "JetHT_{}_{}_{}".format(runType, alignment, datasetName),
                "dir": workDir,
                "exe": "cmsRun",
                "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/JetHT_cfg.py".format(os.environ["CMSSW_BASE"]),
                "run-mode": "Condor",
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
                    "config": local,
                    "dependencies": [],
                }

                ##Loop over all single jobs and set them dependencies for the merge job
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    singleAlignment, singleDatasetName = singleJob["name"].split("_")[2:]

                    if singleDatasetName in config["validations"]["JetHT"][runType][datasetName]["singles"]:
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
                local["jethtplot"]["lumiPerIovFile"] = "{}/src/Alignment/OfflineValidation/data/lumiPerRun_Run2.txt".format(os.environ["CMSSW_BASE"])

            #Write job info
            job = {
                "name": "JetHT_{}_{}".format(runType, datasetName),
                "dir": workDir,
                "exe": "jetHtPlotter",
                "run-mode": "Condor",
                "config": local,
                "dependencies": [],
            }

            for alignment in config["validations"]["JetHT"][runType][datasetName]["alignments"]:

                ##Loop over all merge jobs and set them dependencies for the plot job
                for mergeJob in mergeJobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    mergeAlignment, mergeDatasetName = mergeJob["name"].split("_")[2:]

                    if mergeDatasetName in config["validations"]["JetHT"][runType][datasetName]["merges"]:
                        job["dependencies"].append(mergeJob["name"])

                plotJobs.append(job)

        jobs.extend(plotJobs)
        
    return jobs
