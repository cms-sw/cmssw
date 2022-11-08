import copy
import os

def SplitV(config, validationDir):
    ##List with all jobs
    jobs = []
    SplitVType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single SplitV jobs
    if not SplitVType in config["validations"]["SplitV"]: 
        raise Exception("No 'single' key word in config for SplitV") 

    for singleName in config["validations"]["SplitV"][SplitVType]:
        for IOV in config["validations"]["SplitV"][SplitVType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["SplitV"][SplitVType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/SplitV/{}/{}/{}/{}".format(validationDir, SplitVType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/SplitV/{}/{}/{}/{}".format(config["LFS"], config["name"], SplitVType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["SplitV"][SplitVType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "SplitV_{}_{}_{}_{}".format(SplitVType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/SplitV_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                jobs.append(job)

    ##Do merge SplitV if wished
    if "merge" in config["validations"]["SplitV"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        SplitVType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["SplitV"][SplitVType]:
            for IOV in IOVs:
                ##Work directory for each IOV
                workDir = "{}/SplitV/{}/{}/{}".format(validationDir, SplitVType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "SplitV_{}_{}_{}".format(SplitVType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "SplitVmerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    if alignment in config["validations"]["SplitV"]["single"][mergeName]["alignments"]:
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["SplitV"][SplitVType][mergeName])
                local["output"] = "{}/{}/SplitV/{}/{}/{}".format(config["LFS"], config["name"], SplitVType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]

                    if int(singleIOV) == IOV and singleName in config["validations"]["SplitV"][SplitVType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    return jobs
