import copy
import os
import pprint

def PV(config, validationDir):
    ##List with all jobs
    jobs, singleJobs = [], [] 
    pvType = "single"

    ##Dictionary of lists of all IOVs (can be different per each single job)
    IOVs = {}

    ##Auxilliary dictionary of isData flags per each merged job
    isDataMerged = {}

    ##Start with single PV jobs
    if not pvType in config["validations"]["PV"]: 
        raise Exception("No 'single' key word in config for PV") 

    for singleName in config["validations"]["PV"][pvType]:
        #print("Reading singleName = {}".format(singleName))
        aux_IOV = config["validations"]["PV"][pvType][singleName]["IOV"]
        if not isinstance(aux_IOV, list) and aux_IOV.endswith(".txt"):
            config["validations"]["PV"][pvType][singleName]["IOV"] = []
            with open(aux_IOV, 'r') as IOVfile:
                for line in IOVfile.readlines():
                    if len(line) != 0: config["validations"]["PV"][pvType][singleName]["IOV"].append(int(line))
        for IOV in config["validations"]["PV"][pvType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if singleName not in IOVs.keys():
                IOVs[singleName] = []
            if IOV not in IOVs[singleName]:
                IOVs[singleName].append(IOV) 

            for alignment in config["validations"]["PV"][pvType][singleName]["alignments"]:
                ##Work directory for each IOV
                workDir = "{}/PV/{}/{}/{}/{}".format(validationDir, pvType, singleName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = "{}/{}/PV/{}/{}/{}/{}".format(config["LFS"], config["name"], pvType, alignment, singleName, IOV)
                local["alignment"] = copy.deepcopy(config["alignments"][alignment])
                local["alignment"]["name"] = alignment
                local["validation"] = copy.deepcopy(config["validations"]["PV"][pvType][singleName])
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "name": "PV_{}_{}_{}_{}".format(pvType, alignment, singleName, IOV),
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
            ##Search for MC single(s)
            singlesMC = []
            for singleName in config["validations"]["PV"][pvType][mergeName]['singles']:
                if len(IOVs[singleName]) == 1 and int(IOVs[singleName][0]) == 1: singlesMC.append(singleName)
            isMConly = (len(singlesMC) == len(config["validations"]["PV"][pvType][mergeName]['singles']))
            if isMConly:
                isDataMerged[mergeName] = 0
            elif len(singlesMC) == 0:
                isDataMerged[mergeName] = 1
            else:
                isDataMerged[mergeName] = -1 

            ##Loop over singles
            for iname,singleName in enumerate(config["validations"]["PV"][pvType][mergeName]['singles']):
                isMC = (singleName in singlesMC)
                if isMConly and iname > 0: continue #special case for MC only comparison
                elif isMConly: singlesMC.pop(singlesMC.index(singleName))

                for IOV in IOVs[singleName]:
                    if isMC and not isMConly: continue #ignore IOV=1 as it is automatically added to each DATA IOV unless MC only comparison            
                    #print("mergeName = {}".format(mergeName))

                    ##Work directory for each IOV
                    workDir = "{}/PV/{}/{}/{}".format(validationDir, pvType, mergeName, IOV) #Different (DATA) single jobs must contain different set of IOVs

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

                    ##Deep copy necessary things from global config + assure plot order
                    for alignment in config["alignments"]:
                        idxIncrement = 0
                        local.setdefault("alignments", {})
                        if alignment in config["validations"]["PV"]["single"][singleName]["alignments"]: #Cover all DATA validations
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                            local["alignments"][alignment]['index'] = config["validations"]["PV"]["single"][singleName]["alignments"].index(alignment)
                            local["alignments"][alignment]['isMC'] = False
                        for singleMCname in singlesMC:
                            if alignment in config["validations"]["PV"]["single"][singleMCname]["alignments"]: #Add MC objects
                                local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                                local["alignments"][alignment]['index']  = len(config["validations"]["PV"]["single"][singleName]["alignments"])
                                local["alignments"][alignment]['index'] += idxIncrement + config["validations"]["PV"]["single"][singleMCname]["alignments"].index(alignment)
                                local["alignments"][alignment]['isMC'] = True
                            idxIncrement += len(config["validations"]["PV"]["single"][singleMCname]["alignments"]) 
                    local["validation"] = copy.deepcopy(config["validations"]["PV"][pvType][mergeName])
                    local["validation"]["IOV"] = IOV
                    if "customrighttitle" in local["validation"].keys():
                        if "IOV" in local["validation"]["customrighttitle"]:
                            local["validation"]["customrighttitle"] = local["validation"]["customrighttitle"].replace("IOV",str(IOV)) 
                    local["output"] = "{}/{}/PV/{}/{}/{}".format(config["LFS"], config["name"], pvType, mergeName, IOV)

                    ##Add global plotting options
                    if "style" in config.keys():
                        if "PV" in config['style'].keys():
                            if pvType in config['style']['PV'].keys():
                                local["style"] = copy.deepcopy(config["style"]["PV"][pvType])
                                if "Rlabel" in local["style"] and "customrighttitle" in local["validation"].keys():
                                    print("WARNING: custom right label is overwritten by global settings")

                    ##Loop over all single jobs
                    for singleJob in jobs:
                        ##Get single job info and append to merge job if requirements fullfilled
                        _alignment, _singleName, _singleIOV = singleJob["name"].split("_")[2:]
                        if _singleName in config["validations"]["PV"][pvType][mergeName]["singles"]:
                            if int(_singleIOV) == IOV or (int(_singleIOV) == 1 and _singleName in singlesMC): #matching DATA job or any MC single job 
                                local["alignments"][_alignment]["file"] = singleJob["config"]["output"]
                                job["dependencies"].append(singleJob["name"])
                            
                    mergeJobs.append(job)

        jobs.extend(mergeJobs)

    if "trends" in config["validations"]["PV"]:

        ##List with merge jobs, will be expanded to jobs after looping
        trendJobs = []
        pvType = "trends"

        for trendName in config["validations"]["PV"][pvType]:
            #print("trendName = {}".format(trendName))
            
            ##Work directory for each IOV
            workDir = "{}/PV/{}/{}".format(validationDir, pvType, trendName)

            ##Write general job info
            local = {}

            job = {
                "name": "PV_{}_{}".format(pvType, trendName),
                "dir": workDir,
                "exe": "PVtrends",
                "run-mode": "Condor",
                "dependencies": [],
                "config": local,
            }

            ##Loop over singles
            if 'merges' in config["validations"]["PV"][pvType][trendName].keys()\
               or 'singles' not in config["validations"]["PV"][pvType][trendName].keys():
                raise Exception("Specify list of \'singles\' to run PV trends.")
                #TODO: possible also to run over merges for consistency with DMR jobs
            trendIOVs = [] #TODO: allow different IOV list for each single job? 
            alignmentList = []
            for iname, singleName in enumerate(config["validations"]["PV"][pvType][trendName]["singles"]):
                isMC = (len(IOVs[singleName]) == 1 and int(IOVs[singleName][0]) == 1)
                if isMC:
                    raise Exception("Trend jobs are not implemented for treating MC.")  
                if iname == 0: 
                    trendIOVs = IOVs[singleName] 
                else:
                    for IOV in IOVs[singleName]:
                        if IOV not in trendIOVs or (len(IOVs[singleName]) != len(trendIOVs)):
                            raise Exception("List of IOVs must be the same for each single job.")
                for alignment in config["validations"]["PV"]["single"][singleName]["alignments"]:        
                    if alignment not in alignmentList and alignment in config["alignments"]:
                        local.setdefault("alignments", {})
                        alignmentList.append(alignment)
                        local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                        local["alignments"][alignment]["file"] = "{}/{}/PV/{}/{}/{}/{}".format(config["LFS"], config["name"], "single", alignment, singleName, "{}")
            trendIOVs.sort()
            local["validation"] = copy.deepcopy(config["validations"]["PV"][pvType][trendName])
            local["validation"]["IOV"] = trendIOVs
            if "label" in config["validations"]["PV"][pvType][trendName]:
                local["validation"]["label"] = copy.deepcopy(config["validations"]["PV"][pvType][trendName]["label"])
            local["output"] = "{}/{}/PV/{}/{}/".format(config["LFS"], config["name"], pvType, trendName)
            if "style" in config.keys() and "trends" in config["style"].keys():
                local["style"] = copy.deepcopy(config["style"])
                if "PV" in local["style"].keys(): local["style"].pop("PV")
                if "CMSlabel" in config["style"]["trends"].keys(): local["style"]["CMSlabel"] = config["style"]["trends"]["CMSlabel"]
                if "Rlabel" in config["style"]["trends"].keys():
                    local["style"]["trends"].pop("Rlabel")
                    local["style"]["trends"]["TitleCanvas"] = config["style"]["trends"]["Rlabel"]
            else:
                raise Exception("You want to create 'trends' jobs, but there are no 'lines' section in the config for pixel updates!")

            #Loop over all single jobs
            for singleJob in singleJobs:
                #Get single job info and append to job if requirements fullfilled
                alignment, singleName, singleIOV = singleJob["name"].split("_")[2:]
                
                if singleName in config["validations"]["PV"][pvType][trendName]["singles"]\
                  and int(singleIOV) in trendIOVs:
                    job["dependencies"].append(singleJob["name"])

            trendJobs.append(job)

        jobs.extend(trendJobs)

    return jobs
