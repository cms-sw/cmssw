import copy
import os

def Zmumu(config, validationDir):
    ##List with all jobs
    jobs = []
    zmumuType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single Zmumu jobs
    if not zmumuType in config["validations"]["Zmumu"]: 
        raise Exception("No 'single' key word in config for Zmumu") 

    for datasetName in config["validations"]["Zmumu"][zmumuType]:
        for IOV in config["validations"]["Zmumu"][zmumuType][datasetName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["Zmumu"][zmumuType][datasetName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/Zmumu/{}/{}/{}/{}".format(validationDir, zmumuType, datasetName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/{}/{}/{}/{}".format(config["LFS"], config["name"], zmumuType, alignment, datasetName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["Zmumu"][zmumuType][datasetName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "Zmumu_{}_{}_{}_{}".format(zmumuType, alignment, datasetName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/Zmumu_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                jobs.append(job)

    ##Do merge Zmumu if wished
    if "merge" in config["validations"]["Zmumu"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        zmumuType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["Zmumu"][zmumuType]:
            for IOV in IOVs:
                ##Work directory for each IOV
                workDir = "{}/Zmumu/{}/{}/{}".format(validationDir, zmumuType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "Zmumu_{}_{}_{}".format(zmumuType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "Zmumumerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                    local["validation"] = copy.deepcopy(config["validations"]["Zmumu"][zmumuType][mergeName])
                    local["output"] = "{}/{}/{}/{}/{}".format(config["LFS"], config["name"], zmumuType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, datasetName, singleIOV = singleJob["name"].split("_")[2:]    

                    if int(singleIOV) == IOV and datasetName in config["validations"]["Zmumu"][zmumuType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    return jobs
