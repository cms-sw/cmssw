import copy
import os

def Zmumu(config, validationDir):
    ##List with all jobs
    jobs, singleJobs = [], []
    zmumuType = "single"

    ##Dictionary of lists of all IOVs (can be different per each single job)
    IOVs = {}

    ##Auxilliary dictionary of isData flags per each merged job
    isDataMerged = {}

    ##Start with single Zmumu jobs
    if not zmumuType in config["validations"]["Zmumu"]: 
        raise Exception("No 'single' key word in config for Zmumu") 

    for singleName in config["validations"]["Zmumu"][zmumuType]:
        aux_IOV = config["validations"]["Zmumu"][zmumuType][singleName]["IOV"]
        if not isinstance(aux_IOV, list) and aux_IOV.endswith(".txt"):
            config["validations"]["Zmumu"][zmumuType][singleName]["IOV"] = []
            with open(aux_IOV, 'r') as IOVfile:
                for line in IOVfile.readlines():
                    if len(line) != 0: config["validations"]["Zmumu"][zmumuType][singleName]["IOV"].append(int(line))
        for IOV in config["validations"]["Zmumu"][zmumuType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if singleName not in IOVs.keys():
                IOVs[singleName] = []
            if IOV not in IOVs[singleName]:
                IOVs[singleName].append(IOV)

            for alignment in config["validations"]["Zmumu"][zmumuType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/Zmumu/{}/{}/{}/{}".format(validationDir, zmumuType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/Zmumu/{}/{}/{}/{}".format(config["LFS"], config["name"], zmumuType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["alignment"]["name"] = alignment
                local["validation"] = copy.deepcopy(config["validations"]["Zmumu"][zmumuType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "Zmumu_{}_{}_{}_{}".format(zmumuType, alignment, singleName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/Zmumu_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local, 
                }

                singleJobs.append(job)

    jobs.extend(singleJobs)            

    ##Do merge Zmumu if wished
    if "merge" in config["validations"]["Zmumu"]:
        ##List with merge jobs, will be expanded to jobs after looping
        mergeJobs = []
        zmumuType = "merge"

        ##Loop over all merge jobs/IOVs which are wished
        for mergeName in config["validations"]["Zmumu"][zmumuType]:
            ##Search for MC single(s)
            singlesMC = []
            for singleName in config["validations"]["Zmumu"][zmumuType][mergeName]['singles']:
                if len(IOVs[singleName]) == 1 and int(IOVs[singleName][0]) == 1: singlesMC.append(singleName)
            isMConly = (len(singlesMC) == len(config["validations"]["Zmumu"][zmumuType][mergeName]['singles']))
            if isMConly:
                isDataMerged[mergeName] = 0
            elif len(singlesMC) == 0:
                isDataMerged[mergeName] = 1
            else:
                isDataMerged[mergeName] = -1

            ##Loop over singles
            for iname,singleName in enumerate(config["validations"]["Zmumu"][zmumuType][mergeName]['singles']):
                isMC = (singleName in singlesMC)
                if isMConly and iname > 0: continue #special case for MC only comparison
                elif isMConly: singlesMC.pop(singlesMC.index(singleName))

                for IOV in IOVs[singleName]:
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

                    ##Deep copy necessary things from global config + assure plot order
                    for alignment in config["alignments"]:
                        idxIncrement = 0
                        local.setdefault("alignments", {})
                        if alignment in config["validations"]["Zmumu"]["single"][singleName]["alignments"]: #Cover all DATA validations
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                            local["alignments"][alignment]['index'] = config["validations"]["Zmumu"]["single"][singleName]["alignments"].index(alignment)
                            local["alignments"][alignment]['isMC'] = False
                        for singleMCname in singlesMC:
                            if alignment in config["validations"]["Zmumu"]["single"][singleMCname]["alignments"]: #Add MC objects
                                local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                                local["alignments"][alignment]['index']  = len(config["validations"]["Zmumu"]["single"][singleName]["alignments"])
                                local["alignments"][alignment]['index'] += idxIncrement + config["validations"]["Zmumu"]["single"][singleMCname]["alignments"].index(alignment)
                                local["alignments"][alignment]['isMC'] = True
                            idxIncrement += len(config["validations"]["Zmumu"]["single"][singleMCname]["alignments"])    
                    local["validation"] = copy.deepcopy(config["validations"]["Zmumu"][zmumuType][mergeName])
                    local["validation"]["IOV"] = IOV
                    if "customrighttitle" in local["validation"].keys():
                        if "IOV" in local["validation"]["customrighttitle"]:
                            local["validation"]["customrighttitle"] = local["validation"]["customrighttitle"].replace("IOV",str(IOV))
                    local["output"] = "{}/{}/Zmumu/{}/{}/{}".format(config["LFS"], config["name"], zmumuType, mergeName, IOV) 

                    ##Add global plotting options
                    if "style" in config.keys():
                        if "Zmumu" in config['style'].keys():
                            if zmumuType in config['style']['Zmumu'].keys():
                                local["style"] = copy.deepcopy(config["style"]["Zmumu"][zmumuType])
                                if "Rlabel" in local["style"] and "customrighttitle" in local["validation"].keys():
                                    print("WARNING: custom right label is overwritten by global settings")

                    ##Loop over all single jobs
                    for singleJob in jobs:
                        ##Get single job info and append to merge job if requirements fullfilled
                        _alignment, _singleName, _singleIOV = singleJob["name"].split("_")[2:]
                        if _singleName in config["validations"]["Zmumu"][zmumuType][mergeName]["singles"]:
                            if int(_singleIOV) == IOV or (int(_singleIOV) == 1 and _singleName in singlesMC): #matching DATA job or any MC single job 
                                local["alignments"][_alignment]["file"] = singleJob["config"]["output"]
                                job["dependencies"].append(singleJob["name"])

                    mergeJobs.append(job)  

        jobs.extend(mergeJobs)

    if "trends" in config["validations"]["Zmumu"]:
        print("[WARNING] Zmumu trends are not implemented yet. Nothing to do here...")

    return jobs
-- dummy change --
