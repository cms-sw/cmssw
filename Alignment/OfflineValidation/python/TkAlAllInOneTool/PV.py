import copy
import os
import pprint

def PV(config, validationDir):
    ##List with all jobs
    #jobs = []
    jobs, singleJobs = [], []
    #singleJobs = []
    PVType = "single"

    ##List with all wished IOVs
    IOVs = []

    ##Start with single PV jobs
    if not PVType in config["validations"]["PV"]: 
        raise Exception("No 'single' key word in config for PV") 

    for singleName in config["validations"]["PV"][PVType]:
        print("Reading singleName = {}".format(singleName))
        for IOV in config["validations"]["PV"][PVType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if not IOV in IOVs:
                IOVs.append(IOV)

            for alignment in config["validations"]["PV"][PVType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/PV/{}/{}/{}/{}".format(validationDir, PVType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/PV/{}/{}/{}/{}".format(config["LFS"], config["name"], PVType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["alignment"]["name"] = alignment
                local["validation"] = copy.deepcopy(config["validations"]["PV"][PVType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "PV_{}_{}_{}_{}".format(PVType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/PV_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                #jobs.append(job)
                singleJobs.append(job)

    jobs.extend(singleJobs)

    ##Do merge PV if wished
    if "merge" in config["validations"]["PV"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        pvType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["PV"][pvType]:
            for IOV in IOVs:
                print("mergeName = {}".format(mergeName))
                ##Work directory for each IOV
                workDir = "{}/PV/{}/{}/{}".format(validationDir, pvType, mergeName, IOV)

                ##Write job info
                local = {}

                job = {
                    "name": "PV_{}_{}_{}".format(pvType, mergeName, IOV),
                    "dir": workDir,
                    "exe": "PVmerge",
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                for alignment in config["alignments"]:
                    ##Deep copy necessary things from global config
                    local.setdefault("alignments", {})
                    if alignment in config["validations"]["PV"]["single"][mergeName]["alignments"]:
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])

                local["validation"] = copy.deepcopy(config["validations"]["PV"][pvType][mergeName])
                local["validation"]["IOV"] = IOV
                local["output"] = "{}/{}/PV/{}/{}/{}".format(config["LFS"], config["name"], pvType, mergeName, IOV)

                ##Loop over all single jobs
                for singleJob in jobs:
                    ##Get single job info and append to merge job if requirements fullfilled
                    alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]

                    if int(singleIOV) == IOV and singleName in config["validations"]["PV"][pvType][mergeName]["singles"]:
                        local["alignments"][alignment]["file"] = singleJob["config"]["output"]
                        job["dependencies"].append(singleJob["name"])
                        
                mergeJobs.append(job)

        jobs.extend(mergeJobs)

    if "trends" in config["validations"]["PV"]:

        ##List with merge jobs, will be expanded to jobs after looping
        trendJobs = []
        pvType = "trends"

        for trendName in config["validations"]["PV"][pvType]:
            print("trendName = {}".format(trendName))
            ##Work directory for each IOV
            workDir = "{}/PV/{}/{}".format(validationDir, pvType, trendName)

            ##Write job info
            local = {}

            job = {
                "name": "PV_{}_{}".format(pvType, trendName),
                "dir": workDir,
                "exe": "PVtrends",
                "run-mode": "Condor",
                "dependencies": [],
                "config": local,
            }

            for alignment in config["alignments"]:
                ##Deep copy necessary things from global config
                local.setdefault("alignments", {})
                if alignment in config["validations"]["PV"]["single"][trendName]["alignments"]:
                    local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                    local["alignments"][alignment]["file"] = "{}/{}/PV/{}/{}/{}/{}".format(config["LFS"], config["name"], "single", alignment, trendName, "{}")
            local["validation"] = copy.deepcopy(config["validations"]["PV"][pvType][trendName])
            local["validation"]["IOV"] = IOVs
            if "label" in config["validations"]["PV"][pvType][trendName]:
                local["validation"]["label"] = copy.deepcopy(config["validations"]["PV"][pvType][trendName]["label"])
            local["output"] = "{}/{}/PV/{}/{}/".format(config["LFS"], config["name"], pvType, trendName)
            if config["style"]:
                local["style"] = copy.deepcopy(config["style"])
            else:
                raise Exception("You want to create 'trends' jobs, but there are no 'lines' section in the config for pixel updates!")

            #Loop over all single jobs
            for singleJob in singleJobs:
                #Get single job info and append to job if requirements fullfilled
                alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]
                
                if singleName in config["validations"]["PV"][pvType][trendName]["singles"]:
                    job["dependencies"].append(singleJob["name"])

            trendJobs.append(job)

        jobs.extend(trendJobs)

    return jobs
