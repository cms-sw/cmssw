import copy
import os

def DMR(config, validationDir):
    ##List with all jobs
    jobs = []
    dmrType = "single"

    ##Dictionary of lists of all IOVs (can be different per each single job)
    IOVs = {}

    ##Auxilliary dictionary of isData flags per each merged job
    isDataMerged = {} 

    ##Start with single DMR jobs
    if not dmrType in config["validations"]["DMR"]: 
        raise Exception("No 'single' key word in config for DMR") 

    for singleName in config["validations"]["DMR"][dmrType]:
        aux_IOV = config["validations"]["DMR"][dmrType][singleName]["IOV"]
        if not isinstance(aux_IOV, list) and aux_IOV.endswith(".txt"):
            config["validations"]["DMR"][dmrType][singleName]["IOV"] = []
            with open(aux_IOV, 'r') as IOVfile:
                for line in IOVfile.readlines():
                    if len(line) != 0: config["validations"]["DMR"][dmrType][singleName]["IOV"].append(int(line))
        for IOV in config["validations"]["DMR"][dmrType][singleName]["IOV"]:
            ##Save IOV to loop later for merge jobs
            if singleName not in IOVs.keys():
                IOVs[singleName] = [] 
            if IOV not in IOVs[singleName]:
                IOVs[singleName].append(IOV)

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

        ##Loop over all merge jobs
        for mergeName in config["validations"]["DMR"][dmrType]:
            ##Search for MC single(s)
            singlesMC = []
            for singleName in config["validations"]["DMR"][dmrType][mergeName]['singles']:
                if len(IOVs[singleName]) == 1 and int(IOVs[singleName][0]) == 1: singlesMC.append(singleName) 
            isMConly = (len(singlesMC) == len(config["validations"]["DMR"][dmrType][mergeName]['singles']))
            if isMConly:
                isDataMerged[mergeName] = 0
            elif len(singlesMC) == 0:
                isDataMerged[mergeName] = 1
            else:
                isDataMerged[mergeName] = -1   

            ##Loop over singles 
            for iname,singleName in enumerate(config["validations"]["DMR"][dmrType][mergeName]['singles']):
                isMC = (singleName in singlesMC)
                if isMConly and iname > 0: continue #special case for MC only comparison
                elif isMConly: singlesMC.pop(singlesMC.index(singleName))
  
                for IOV in IOVs[singleName]:
                    if isMC and not isMConly: continue #ignore IOV=1 as it is automatically added to each DATA IOV unless MC only comparison
 
                    ##Work directory for each IOV
                    workDir = "{}/DMR/{}/{}/{}".format(validationDir, dmrType, mergeName, IOV) #Different (DATA) single jobs must contain different set of IOVs

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

                    ##Deep copy necessary things from global config + assure plot order
                    for alignment in config["alignments"]:
                        idxIncrement = 0
                        local.setdefault("alignments", {})
                        if alignment in config["validations"]["DMR"]["single"][singleName]["alignments"]: #Cover all DATA validations
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                            local["alignments"][alignment]['index'] = config["validations"]["DMR"]["single"][singleName]["alignments"].index(alignment)
                        for singleMCname in singlesMC:
                            if alignment in config["validations"]["DMR"]["single"][singleMCname]["alignments"]: #Add MC objects
                                local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                                local["alignments"][alignment]['index']  = len(config["validations"]["DMR"]["single"][singleName]["alignments"])
                                local["alignments"][alignment]['index'] += idxIncrement + config["validations"]["DMR"]["single"][singleMCname]["alignments"].index(alignment)                           
                            idxIncrement += len(config["validations"]["DMR"]["single"][singleMCname]["alignments"])   
                    local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][mergeName])
                    local["validation"]["IOV"] = IOV #is it really needed here?
                    if "customrighttitle" in local["validation"].keys():
                        if "IOV" in local["validation"]["customrighttitle"]:
                            local["validation"]["customrighttitle"] = local["validation"]["customrighttitle"].replace("IOV",str(IOV)) 
                    local["output"] = "{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], dmrType, mergeName, IOV)

                    ##Add global plotting options
                    if "style" in config.keys():
                        if "DMR" in config['style'].keys():
                            if dmrType in config['style']['DMR'].keys():
                                local["style"] = copy.deepcopy(config["style"]["DMR"][dmrType])
                                if "Rlabel" in local["style"] and "customrighttitle" in local["validation"].keys():
                                    print("WARNING: custom right label is overwritten by global settings") 
 
                    ##Loop over all single jobs
                    for singleJob in jobs:
                        ##Get single job info and append to merge job if requirements fullfilled
                        _alignment, _singleName, _singleIOV = singleJob["name"].split("_")[2:]
                        if _singleName in config["validations"]["DMR"][dmrType][mergeName]["singles"]:
                            if int(_singleIOV) == IOV or (int(_singleIOV) == 1 and _singleName in singlesMC): #matching DATA job or any MC single job 
                                local["alignments"][_alignment]["file"] = singleJob["config"]["output"]
                                job["dependencies"].append(singleJob["name"])
                                
                    ##Append to merge jobs  
                    mergeJobs.append(job)

        ##Append to all jobs
        jobs.extend(mergeJobs)

    if "trends" in config["validations"]["DMR"]:

        ##List with merge jobs, will be expanded to jobs after looping
        trendJobs = []
        dmrType = "trends"

        for trendName in config["validations"]["DMR"][dmrType]:
            #print("trendName = {}".format(trendName))
            ##Work directory for each IOV
            workDir = "{}/DMR/{}/{}".format(validationDir, dmrType, trendName)
 
            ##Write general job info
            local = {}
            job = {
                "name": "DMR_{}_{}".format(dmrType, trendName),
                "dir": workDir,
                "exe": "DMRtrends",
                "run-mode": "Condor",
                "dependencies": [],
                "config": local,
            }

            ###Loop over merge steps (merge step can contain only DATA)
            mergesDATA = []
            for mergeName in config["validations"]["DMR"][dmrType][trendName]["merges"]:
                ##Validate merge step
                if isDataMerged[mergeName] < 0:
                    raise Exception("Trend jobs cannot process merge jobs containing both DATA and MC objects.")
                elif isDataMerged[mergeName] == 1:
                    mergesDATA.append(mergeName)
                else:
                    if "doUnitTest" in config["validations"]["DMR"][dmrType][trendName].keys() and config["validations"]["DMR"][dmrType][trendName]["doUnitTest"]:
                        local.setdefault("alignments", {})
                        continue
                    else: 
                        raise Exception("Trend jobs are not implemented for treating MC.")

            ###Loop over DATA singles included in merge steps
            trendIOVs = []
            _mergeFiles = []
            for mergeName in mergesDATA:
                for iname,singleName in enumerate(config["validations"]["DMR"]['merge'][mergeName]['singles']): 
                    trendIOVs += [IOV for IOV in IOVs[singleName]]
                    ##Deep copy necessary things from global config + ensure plot order
                    for alignment in config["alignments"]:
                        local.setdefault("alignments", {})
                        if alignment in config["validations"]["DMR"]["single"][singleName]["alignments"]: #Cover all DATA validations
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                            local["alignments"][alignment]['index'] = config["validations"]["DMR"]["single"][singleName]["alignments"].index(alignment)
                _mergeFiles.append("{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], "merge", mergeName, "{}")) 
            trendIOVs.sort() 
            local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][trendName])
            if len(_mergeFiles) == 1:
                local["validation"]["mergeFile"] = _mergeFiles[0]
            else:
                local["validation"]["mergeFile"] = _mergeFiles #FIXME for multiple merge files in backend
            local["validation"]["IOV"] = trendIOVs
            local["output"] = "{}/{}/DMR/{}/{}/".format(config["LFS"], config["name"], dmrType, trendName)
            if "style" in config.keys() and "trends" in config["style"].keys():
                local["style"] = copy.deepcopy(config["style"])
                if "DMR" in local["style"].keys(): local["style"].pop("DMR") 
                if "CMSlabel" in config["style"]["trends"].keys(): local["style"]["CMSlabel"] = config["style"]["trends"]["CMSlabel"]
                if "Rlabel" in config["style"]["trends"].keys(): 
                    local["style"]["trends"].pop("Rlabel")
                    local["style"]["trends"]["TitleCanvas"] = config["style"]["trends"]["Rlabel"]
            else:
                raise Exception("You want to create 'trends' jobs, but there are no 'lines' section in the config for pixel updates!")

            #Loop over all merge jobs
            for mergeName in mergesDATA:
                for mergeJob in mergeJobs:
                    alignment, mergeJobName, mergeIOV = mergeJob["name"].split("_")[1:]
                    if mergeJobName == mergeName and int(mergeIOV) in trendIOVs:
                        job["dependencies"].append(mergeJob["name"])

            trendJobs.append(job)

        jobs.extend(trendJobs)

    if "averaged" in config["validations"]["DMR"]:

        ####Finally, list of jobs is expanded for luminosity-averaged plot (avp) job
        avpJobs = []
        dmrType = "averaged"
        for avpName in config["validations"]["DMR"][dmrType]:
            ###Main workdir for each combination of displayed lumi-averaged validation objects to be plotted
            workDir = "{}/DMR/{}/{}".format(validationDir, dmrType, avpName) 
            output  = "{}/{}/DMR/{}/{}".format(config["LFS"], config["name"], dmrType, avpName)
            
            ###Loop over merge steps (one merge step can contain only DATA or only MC singles but not mix)
            mergesDATA = []
            mergesMC   = []
            for mergeName in config["validations"]["DMR"][dmrType][avpName]["merges"]:
                ##Validate merge step
                if isDataMerged[mergeName] < 0:
                    raise Exception("Average jobs cannot process merge jobs containing both DATA and MC objects.")
                elif isDataMerged[mergeName] == 1: 
                    mergesDATA.append(mergeName)
                else:
                    mergesMC.append(mergeName) 

            lumiPerRun = []
            lumiPerIoV = []
            lumiMC     = [] 
            if len(mergesDATA) > 0:  
                if "lumiPerRun" in config["validations"]["DMR"][dmrType][avpName].keys(): 
                    for lumifile in config["validations"]["DMR"][dmrType][avpName]['lumiPerRun']:
                        if lumifile.split(".")[-1] in ["txt","csv"]:
                            lumiPerRun.append(lumifile)
                if "lumiPerIoV" in config["validations"]["DMR"][dmrType][avpName].keys():
                    for lumifile in config["validations"]["DMR"][dmrType][avpName]['lumiPerIoV']:
                        if lumifile.split(".")[-1] in ["txt","csv"]:
                            lumiPerIoV.append(lumifile)
                if len(lumiPerRun) == 0 and len(lumiPerIoV) == 0: 
                    raise Exception("No lumi per run/IoV file found or not specified in .csv/.txt format.")
            if len(mergesMC) > 0: 
                if 'lumiMC' in config["validations"]["DMR"][dmrType][avpName].keys():
                    lumiMC = config["validations"]["DMR"][dmrType][avpName]['lumiMC']
 
            ###Store information about plotting job in this dictionary
            plotJob = {}
            plotJob['workdir'] = "{}/{}".format(workDir,"plots")
            plotJob['output'] = "{}/{}".format(output,"plots")
            plotJob['inputData'] = []
            plotJob['inputMC'] = []
            plotJob['dependencies'] = []

            ###First loop over DATA 
            for mergeName in mergesDATA:
                ##Adapt workdir per merge step
                workDirMerge = "{}/{}".format(workDir, mergeName)
                outputMerge  = "{}/{}".format(output, mergeName)

                ##Create local config per merge step
                local = {}
                local["type"]   = "DMR"
                local["mode"]   = "merge"
                local["isData"] = True
                local["isMC"]   = False

                ##Deep copy necessary things from global config
                for alignment in config["alignments"]:
                    local.setdefault("alignments", {})
                    for singleName in config["validations"]["DMR"]["merge"][mergeName]["singles"]:
                        if alignment in config["validations"]["DMR"]["single"][singleName]['alignments']:
                            local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])        
                local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][avpName])
                local["validation"]["mergeFile"] = "{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], "merge", mergeName, "{}")
                local["validation"]["lumiPerRun"] = lumiPerRun
                local["validation"]["lumiPerIoV"] = lumiPerIoV
                local["validation"]["lumiMC"]     = lumiMC
                local["validation"]["firstFromNext"] = [] 

                ##Determine list of IOVs in merge step (generally different than total list of IOVs)
                IOVsPerMergeStep = []
                for singleName in config["validations"]["DMR"]["merge"][mergeName]["singles"]:
                    for IOV in IOVs[singleName]: 
                        if IOV not in IOVsPerMergeStep:
                            IOVsPerMergeStep.append(IOV) 
                IOVsPerMergeStep.sort()
 
                ##Divide average step job to subjobs to prevent memory issues
                extra_part = 0
                maxfiles = int(config["validations"]["DMR"][dmrType][avpName]['maxfiles'])
                if len(IOVsPerMergeStep)%maxfiles >= 2:
                    extra_part = 1
                parts = extra_part+len(IOVsPerMergeStep)//maxfiles

                subJob = {'name' : [], 'output' : [], 'lumiPerFile' : []}
                for ipart in range(0,parts):
                    #Adapt workdir per each subjob
                    workDirSub = workDirMerge+"_"+str(ipart)
                    outputSub = outputMerge+"_"+str(ipart)  

                    #Define IOV group  
                    IOVGroup = []
                    lastIndex = 0
                    for iIOV,IOV in enumerate(IOVsPerMergeStep):
                        if (iIOV//maxfiles == ipart) or (ipart == parts-1 and iIOV//maxfiles > ipart):
                            IOVGroup.append(IOV)
                            lastIndex = iIOV
                    firstFromNext = []
                    if lastIndex != len(IOVsPerMergeStep)-1:
                        firstFromNext.append(IOVsPerMergeStep[lastIndex+1]) 

                    #Write job info
                    _local = copy.deepcopy(local)
                    _local["output"] = outputSub
                    _local["validation"]["IOV"] = IOVGroup 
                    _local["validation"]["firstFromNext"] = firstFromNext
                    job = {
                        "name": "DMR_{}_{}_{}".format(dmrType, avpName, mergeName+"_"+str(ipart)),
                        "dir": workDirSub,
                        "exe": "mkLumiAveragedPlots.py",
                        "run-mode": "Condor",
                        "dependencies": [],
                        "config": _local,
                    } 
                    subJob['output'].append(outputSub)    
                    subJob['name'].append("DMR_{}_{}_{}".format(dmrType, avpName, mergeName+"_"+str(ipart)))
                    subJob['lumiPerFile'].append(os.path.join(outputSub,"lumiPerFile.csv"))                
                    if parts == 1:
                        plotJob['inputData'].append(outputSub)  
                        plotJob['dependencies'].append(job['name'])

                    #Set average job dependencies from the list of all merge jobs
                    for mergeJob in mergeJobs:
                        alignment, mergeJobName, mergeIOV = mergeJob["name"].split("_")[1:]
                        if mergeJobName == mergeName and int(mergeIOV) in IOVGroup:
                            job["dependencies"].append(mergeJob["name"]) 
                        #if mergeJobName in config["validations"]["DMR"][dmrType][avpName]["merges"]:
                        #    job["dependencies"].append(mergeJob["name"])

                    #Add to queue 
                    avpJobs.append(job) 

                ##Add finalization job to merge all subjobs 
                if parts > 1:
                    localFinalize = copy.deepcopy(local)
                    localFinalize['mode'] = "finalize"
                    localFinalize['output'] = outputMerge
                    localFinalize["validation"]["IOV"] = []
                    localFinalize["validation"]["mergeFile"] = subJob['output'] 
                    localFinalize["validation"]["lumiPerRun"] = []
                    localFinalize["validation"]["lumiPerIoV"] = subJob['lumiPerFile']
                    job = {
                    "name": "DMR_{}_{}_{}".format(dmrType, avpName, mergeName+"_finalize"),
                    "dir": workDirMerge,
                    "exe": "mkLumiAveragedPlots.py",
                    "run-mode": "Condor",
                    "dependencies": subJob['name'],
                    "config": localFinalize,
                    }
                    avpJobs.append(job)
                    plotJob['inputData'].append(outputMerge)
                    plotJob['dependencies'].append(job['name']) 

            #Second create one averager job per all MC merge jobs
            if len(mergesMC) != 0: 
                ##Adapt workdir per merge step
                workDirMerge = "{}/{}".format(workDir, "MC")
                outputMerge  = "{}/{}".format(output, "MC")

                ##Create local config for MC average job
                local = {}
                local["type"]   = "DMR"
                local["mode"]   = "merge" 
                local["isData"] = False
                local["isMC"] = True
                local["output"] = outputMerge        

                ##Deep copy necessary things from global config
                local["validation"] = copy.deepcopy(config["validations"]["DMR"][dmrType][avpName])
                local["validation"]["mergeFile"] = []
                for mergeName in mergesMC:
                    for alignment in config["alignments"]:
                        local.setdefault("alignments", {})
                        for singleName in config["validations"]["DMR"]["merge"][mergeName]["singles"]:
                            if alignment in config["validations"]["DMR"]["single"][singleName]['alignments']:
                                local["alignments"][alignment] = copy.deepcopy(config["alignments"][alignment])
                    local["validation"]["mergeFile"].append("{}/{}/DMR/{}/{}/{}".format(config["LFS"], config["name"], "merge", mergeName, "{}"))
                local["validation"]["lumiPerRun"] = lumiPerRun
                local["validation"]["lumiPerIoV"] = lumiPerIoV
                local["validation"]["lumiMC"]     = lumiMC 
                local["validation"]["IOV"] = [1]

                ##Write job info
                job = {
                            "name": "DMR_{}_{}_{}".format(dmrType, avpName, mergeName+"_MC"),
                            "dir": workDirMerge,
                            "exe": "mkLumiAveragedPlots.py",
                            "run-mode": "Condor",
                            "dependencies": [],
                            "config": local,
                        }
                plotJob['inputMC'].append(outputMerge)
                plotJob['dependencies'].append(job['name']) 

                ##Set average job dependencies from the list of all merge jobs
                for mergeJob in mergeJobs:
                    alignment, mergeJobName, mergeIOV = mergeJob["name"].split("_")[1:]
                    if mergeJobName in mergesMC:
                        job["dependencies"].append(mergeJob["name"])

                ##Add to queue 
                avpJobs.append(job) 
                  
            ##Finally add job to plot averaged distributions
            if len(plotJob['inputData'])+len(plotJob['inputMC']) > 0:
                local = {}
                local["type"] = "DMR"
                local["mode"]   = "plot"
                local["isData"] = True if len(plotJob['inputData']) > 0 else False
                local["isMC"] = True if len(plotJob['inputMC']) > 0 else False
                local["output"] = plotJob['output']
                local["plot"] = { "inputData" : plotJob['inputData'], 
                                  "inputMC" : plotJob['inputMC'],
                                  "alignments" : [],
                                  "objects" : [],
                                  "labels"  : [],
                                  "colors"  : [],
                                  "styles"  : [],
                                  "useFit"        : True,
                                  "useFitError"   : False,
                                  "showMean"      : True,
                                  "showMeanError" : False,
                                  "showRMS"       : False,
                                  "showRMSError"  : False}
                ##Copy alignment objects info from global config
                for mergeName in mergesDATA+mergesMC:
                    for singleName in config["validations"]["DMR"]["merge"][mergeName]["singles"]:
                        for alignment in config["validations"]["DMR"]["single"][singleName]['alignments']:
                            if alignment in config['alignments'] and alignment not in local["plot"]["alignments"]:
                                local["plot"]["alignments"].append(alignment)
                                objectName = config["alignments"][alignment]["title"].replace(" ","_")
                                if objectName not in local["plot"]["objects"]: #can happen for MC
                                    local["plot"]["objects"].append(objectName)
                                    local["plot"]["labels"].append(config["alignments"][alignment]["title"])
                                    local["plot"]["colors"].append(config["alignments"][alignment]["color"])
                                    local["plot"]["styles"].append(config["alignments"][alignment]["style"])
                ##Overwrite if needed 
                for extraKey in ["objects","labels","colors","styles","useFit","useFitError","showMean","showMeamError","showRMS","showRMSError"]:
                    if extraKey in config["validations"]["DMR"][dmrType][avpName].keys():
                        local["plot"][extraKey] = config["validations"]["DMR"][dmrType][avpName][extraKey]
                ##Add global plotting options
                if "style" in config.keys():
                    if "DMR" in config['style'].keys():
                        if dmrType in config['style']['DMR'].keys():
                            local["plotGlobal"] = copy.deepcopy(config["style"]['DMR'][dmrType])  
                
                ##Write job info 
                job = {
                            "name": "DMR_{}_{}_{}".format(dmrType, avpName, "plot"),
                            "dir": plotJob['workdir'],
                            "exe": "mkLumiAveragedPlots.py",
                            "run-mode": "Condor",
                            "dependencies": plotJob['dependencies'],
                            "config": local,
                        }
                avpJobs.append(job) 

        #Finally extend main job collection
        jobs.extend(avpJobs)

    return jobs
