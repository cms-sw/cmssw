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

    for singleName in config["validations"]["DMR"][dmrType]:
        for IOV in config["validations"]["DMR"][dmrType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["DMR"][dmrType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/DMR/{}/{}/{}/{}".format(validationDir, dmrType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/DMR/{}/{}/{}/{}".format(config["LFS"], config["name"], dmrType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "DMR_{}_{}_{}_{}".format(dmrType, alignment, singleName, IOV),
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
                    if alignment in config["validations"]["DMR"]["single"][mergeName]["alignments"]:
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][mergeName])
                local["validation"]["IOV"] = IOV
                local["output"] = "{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], dmrType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]
                    if int(singleIOV) == IOV and singleName in config["validations"]["DMR"][dmrType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    if "trends" in config["validations"]["DMR"]:

        ##List with merge jobs, will be expanded to jobs after looping
        trendJobs = []
        dmrType = "trends"

        for trendName in config["validations"]["DMR"][dmrType]:
            print("trendName = {}".format(trendName))
            ##Work directory for each IOV
            workDir = "{}/DMR/{}/{}".format(validationDir, dmrType, trendName)

            ##Write job info
            local = {}

            job = {
                "name": "DMR_{}_{}".format(dmrType, trendName),
                "dir": workDir,
                "exe": "DMRtrends",
                "run-mode": "Condor",
                "dependencies": [],
                "config": local,
            }

            for alignment in config["alignments"]:
                ##Deep copy necessary things from global config
                local.setdefault("alignments", {})
                if alignment in config["validations"]["DMR"]["single"][trendName]["alignments"]:
                    local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
            local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][trendName])
            local["validation"]["mergeFile"] = "{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], "merge", trendName, "{}")
            local["validation"]["IOV"] = IOVs
            local["output"] = "{}/{}/DMR/{}/{}/".format(config["LFS"], config["name"], dmrType, trendName)
            if config["lines"]:
                local["lines"] = copy.deepcopy(config["lines"])
            else:
                raise Exception("You want to create 'trends' jobs, but there are no 'lines' section in the config for pixel updates!")

            #Loop over all merge jobs
            for mergeJob in mergeJobs:
                #Get merge job info and append to job if requirements fullfilled
                alignment, mergeName, mergeIOV = mergeJob["name"].split("_")[1:]

                if mergeName in config["validations"]["DMR"][dmrType][trendName]["singles"]:
                    job["dependencies"].append(mergeJob["name"])

            trendJobs.append(job)

        jobs.extend(trendJobs)

    return jobs
