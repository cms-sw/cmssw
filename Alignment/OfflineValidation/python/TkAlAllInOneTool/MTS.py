import copy
import os

def MTS(config, validationDir):
    ##List with all jobs
    jobs = []
    mtsType = "single"

    ##Dictionary of lists of all IOVs (can be different per each single job)
    IOVs = {}

    ##Auxilliary dictionary of isData flags per each merged job
    isDataMerged = {} 

    ##Start with single MTS jobs
    if not mtsType in config["validations"]["MTS"]: 
        raise Exception("No 'single' key word in config for MTS") 

    for singleName in config["validations"]["MTS"][mtsType]:
        aux_IOV = config["validations"]["MTS"][mtsType][singleName]["IOV"]
        if not isinstance(aux_IOV, list) and aux_IOV.endswith(".txt"):
            config["validations"]["MTS"][mtsType][singleName]["IOV"] = []
            with open(aux_IOV, 'r') as IOVfile:
                for line in IOVfile.readlines():
                    if len(line) != 0: config["validations"]["MTS"][mtsType][singleName]["IOV"].append(int(line))
        for IOV in config["validations"]["MTS"][mtsType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if singleName not in IOVs.keys():
                IOVs[singleName] = []
            if IOV not in IOVs[singleName]:
                IOVs[singleName].append(IOV) 
            
            for alignment in config["validations"]["MTS"][mtsType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/MTS/{}/{}/{}/{}".format(validationDir, mtsType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/MTS/{}/{}/{}/{}".format(config["LFS"], config["name"], mtsType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["alignment"]["name"] = alignment
                local["validation"] = copy.deepcopy(config["validations"]["MTS"][mtsType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "MTS_{}_{}_{}_{}".format(mtsType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/MTS_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                jobs.append(job)

    ##Do merge MTS if wished
    if "merge" in config["validations"]["MTS"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        pvType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["MTS"][pvType]:
            ##Loop over singles
            for iname,singleName in enumerate(config["validations"]["MTS"][pvType][mergeName]['singles']):
                for IOV in IOVs[singleName]:
                    
                    ##Work directory for each IOV
                    workDir = "{}/MTS/{}/{}/{}".format(validationDir, pvType, mergeName, IOV) #Different (DATA) single jobs must contain different set of IOVs

                    ##Write job info
                    local = {}

                    job = {
                        "name": "MTS_{}_{}_{}".format(pvType, mergeName, IOV),
                        "dir": workDir,
                        "exe": "MTSmerge",
                        "run-mode": "Condor",
                        "dependencies": [],
                        "config": local, 
                    }

                    ##Deep copy necessary things from global config + assure plot order
                    for alignment in config["alignments"]:
                        local.setdefault("alignments", {})
                        if alignment in config["validations"]["MTS"]["single"][singleName]["alignments"]: #Cover all DATA validations
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                            local["alignments"][alignment]['index'] = config["validations"]["MTS"]["single"][singleName]["alignments"].index(alignment)
                            local["alignments"][alignment]['isMC'] = False
                    local["validation"] = copy.deepcopy(config["validations"]["MTS"][pvType][mergeName])
                    local["validation"]["IOV"] = IOV
                    if "customrighttitle" in local["validation"].keys():
                        if "IOV" in local["validation"]["customrighttitle"]:
                            local["validation"]["customrighttitle"] = local["validation"]["customrighttitle"].replace("IOV",str(IOV)) 
                    local["output"] = "{}/{}/MTS/{}/{}/{}".format(config["LFS"], config["name"], pvType, mergeName, IOV)

                    ##Add global plotting options
                    if "style" in config.keys():
                        if "MTS" in config['style'].keys():
                            if pvType in config['style']['MTS'].keys():
                                local["style"] = copy.deepcopy(config["style"]["MTS"][pvType])
                                if "Rlabel" in local["style"] and "customrighttitle" in local["validation"].keys():
                                    print("WARNING: custom right label is overwritten by global settings")

                    ##Loop over all single jobs
                    for singleJob in jobs:
                        ##Get single job info and append to merge job if requirements fullfilled
                        _alignment, _singleName, _singleIOV = singleJob["name"].split("_")[2:]
                        if _singleName in config["validations"]["MTS"][pvType][mergeName]["singles"]:
                            if (int(_singleIOV) == IOV): #matching DATA job or any MC single job 
                                local["alignments"][_alignment]["file"] = singleJob["config"]["output"]
                                job["dependencies"].append(singleJob["name"])
                            
                    mergeJobs.append(job)

        jobs.extend(mergeJobs)
                
    return jobs
-- dummy change --
