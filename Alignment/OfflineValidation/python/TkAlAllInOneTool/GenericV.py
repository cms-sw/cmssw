import copy
import os

def GenericV(config, validationDir):
    ##List with all jobs
    jobs = []
    GenericVType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single GenericV jobs
    if not GenericVType in config["validations"]["Generic"]: 
        raise Exception("No 'single' key word in config for GenericV") 

    for singleName in config["validations"]["Generic"][GenericVType]:
        for IOV in config["validations"]["Generic"][GenericVType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["Generic"][GenericVType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/GenericV/{}/{}/{}/{}".format(validationDir, GenericVType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/GenericV/{}/{}/{}/{}".format(config["LFS"], config["name"], GenericVType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["Generic"][GenericVType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "GenericV_{}_{}_{}_{}".format(GenericVType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/GenericV_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                jobs.append(job)

    ##Do merge GenericV if wished
    if "merge" in config["validations"]["Generic"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        GenericVType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["Generic"][GenericVType]:
            for IOV in IOVs:
                ##Work directory for each IOV
                workDir = "{}/GenericV/{}/{}/{}".format(validationDir, GenericVType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "GenericV_{}_{}_{}".format(GenericVType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "GenericVmerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    if alignment in config["validations"]["Generic"]["single"][mergeName]["alignments"]:
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["Generic"][GenericVType][mergeName])
                local["output"] = "{}/{}/GenericV/{}/{}/{}".format(config["LFS"], config["name"], GenericVType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]

                    if int(singleIOV) == IOV and singleName in config["validations"]["Generic"][GenericVType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    return jobs
