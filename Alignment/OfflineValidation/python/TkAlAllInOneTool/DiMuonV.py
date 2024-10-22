import copy
import os

def DiMuonV(config, validationDir):
    ##List with all jobs
    jobs = []
    DiMuonVType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single DiMuonV jobs
    if not DiMuonVType in config["validations"]["DiMuonV"]:
        raise Exception("No 'single' key word in config for DiMuonV")

    for singleName in config["validations"]["DiMuonV"][DiMuonVType]:
        for IOV in config["validations"]["DiMuonV"][DiMuonVType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["DiMuonV"][DiMuonVType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/DiMuonV/{}/{}/{}/{}".format(validationDir, DiMuonVType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/DiMuonV/{}/{}/{}/{}".format(config["LFS"], config["name"], DiMuonVType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["DiMuonV"][DiMuonVType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "DiMuonV_{}_{}_{}_{}".format(DiMuonVType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/DiMuonV_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local,
                }

                jobs.append(job)

    ##Do merge DiMuonV if wished
    if "merge" in config["validations"]["DiMuonV"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        DiMuonVType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["DiMuonV"][DiMuonVType]:
            for IOV in IOVs:
                ##Work directory for each IOV
                workDir = "{}/DiMuonV/{}/{}/{}".format(validationDir, DiMuonVType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "DiMuonV_{}_{}_{}".format(DiMuonVType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "DiMuonVmerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local,
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    if alignment in config["validations"]["DiMuonV"]["single"][mergeName]["alignments"]:
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                local["validation"] = copy.deepcopy(config["validations"]["DiMuonV"][DiMuonVType][mergeName])
                local["output"] = "{}/{}/DiMuonV/{}/{}/{}".format(config["LFS"], config["name"], DiMuonVType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]

                    if int(singleIOV) == IOV and singleName in config["validations"]["DiMuonV"][DiMuonVType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])

                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    return jobs
