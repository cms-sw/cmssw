import copy
import os

def DMR(config, validationDir):
    ##List with all jobs
    jobs = []
    dmrType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single DMR jobs
    if not dmrType in config["validations"]["DMR"]: 
        raise Exception("No 'single' key word in config for DMR") 

    for datasetName in config["validations"]["DMR"][dmrType]:
        for IOV in config["validations"]["DMR"][dmrType][datasetName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["DMR"][dmrType][datasetName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/DMR/{}/{}/{}/{}".format(validationDir, dmrType, datasetName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/{}/{}/{}/{}".format(config["LFS"], config["name"], dmrType, alignment, datasetName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][datasetName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "DMR_{}_{}_{}_{}".format(dmrType, alignment, datasetName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/DMR_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                jobs.append(job)

    ##Do merge DMR if wished
    if "merge" in config["validations"]["DMR"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        dmrType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["DMR"][dmrType]:
            for IOV in IOVs:
                ##Work directory for each IOV
                workDir = "{}/DMR/{}/{}/{}".format(validationDir, dmrType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "DMR_{}_{}_{}".format(dmrType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "DMRmerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                    local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][mergeName])
                    local["output"] = "{}/{}/{}/{}/{}".format(config["LFS"], config["name"], dmrType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, datasetName, singleIOV = singleJob["name"].split("_")[2:]    

                    if int(singleIOV) == IOV and datasetName in config["validations"]["DMR"][dmrType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    return jobs
