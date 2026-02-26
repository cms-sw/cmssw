import os
import copy

_validationName = "PixBary"

def PixBary(config, validationDir, verbose=False):
    ##List with all jobs
    jobs = []

    ##Dictionary of lists of all IOVs (can be different per each single job)
    IOVs = {}

    ##Start with single jobs
    jobType = "single"

    ##Common paths
    dir_base = os.path.join(validationDir, _validationName) # Base path for work directories (one for for each job, alignment and IOV)
    out_base = os.path.join(config["LFS"], config["name"], _validationName)

    ##Check that a job is defined
    if not jobType in config["validations"][_validationName]:
        raise LookupError("No '%s' key word in config for %s" %(jobType, _validationName))

    ##Loop over all merge jobs/IOVs which are requested
    for jobName, jobConfig in config["validations"][_validationName][jobType].items():
        IOV_list = get_IOVs(jobConfig)  # The PixelBarycentre automatically detects IOV changes in the payloads. This list is used to specify the run range(s) to analyze
        if(verbose):
            print('job: %s IOV_list: %s', jobName, IOV_list)
        IOVs[jobName] = IOV_list

        ##Loop over IOVs (ranges of runs, in this case)
        for runRange in IOV_list:
            IOV = '-'.join(str(i) for i in runRange)

            for alignment in jobConfig["alignments"]:
                alignmentConfig = config["alignments"][alignment]

                ##Write local config
                local = {}
                local["output"] = os.path.join(out_base, jobType, alignment, jobName, IOV)
                local["alignment"] = copy.deepcopy(alignmentConfig)
                local["alignment"]["label"] = alignment
                local["validation"] = copy.deepcopy(jobConfig)
                local["validation"].pop("alignments")
                local["validation"]["IOV"] = IOV
                if "dataset" in local["validation"]:
                    local["validation"]["dataset"] = local["validation"]["dataset"].format(IOV)
                if "goodlumi" in local["validation"]:
                    local["validation"]["goodlumi"] = local["validation"]["goodlumi"].format(IOV)

                ##Write job info
                job = {
                    "label": jobName, # the name used in the config, so that it can be referenced lated
                    "name": "{}_{}_{}_{}_{}".format(_validationName, alignment, jobType, jobName, IOV),
                    "dir": os.path.join(dir_base, jobType, jobName, alignment, IOV),
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/PixelBaryCentreAnalyzer_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local,
                }

                jobs.append(job)

    # Extract text from the ROOT files
    jobType = "extract"

    for jobName, jobConfig in config["validations"][_validationName][jobType].items():
        for singleName in jobConfig.get("singles"):
            # Search for the "single" job referenced by name
            matchingSingleConfigs = [j for j in jobs if j.get("label", "") == singleName]

            for singleConfig in matchingSingleConfigs:
                IOV = singleConfig["config"]["validation"]["IOV"]  # <str>
                alignment = singleConfig["config"]["alignment"]["label"] # <str>

                job = {
                    "name": "_".join([_validationName, jobType, jobName, singleName, alignment, IOV]),
                    "dir": os.path.join(dir_base, jobType, jobName, singleName, alignment, IOV),
                    "dependencies": [singleConfig["name"]],
                    "exe": "extractBaryCentre.py",
                    "config": {
                        "input": os.path.join(singleConfig["config"]["output"], "PixelBaryCentre.root"),
                        "output": os.path.join(out_base, jobType, jobName, singleName, alignment, IOV),
                        "styles": jobConfig.get("styles", ["csv", "twiki"])
                        },
                    "flavour": "espresso" # So fast that anything else would not make sense
                }
                jobs.append(job)

    return jobs

def get_IOVs(jobConfig):
    return [[jobConfig['firstRun'], jobConfig['lastRun']]]
