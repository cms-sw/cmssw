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

            for alignment, alignmentConfig in config["alignments"].items():
                ##Work directory for each IOV
                workDir = os.path.join(validationDir, _validationName, jobType, jobName, alignment, IOV)

                ##Write local config
                local = {}
                local["output"] = os.path.join(config["LFS"], config["name"], _validationName, jobType, alignment, jobName, IOV)
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
                    "name": "{}_{}_{}_{}_{}".format(_validationName, alignment, jobType, jobName, IOV),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/PixelBaryCentreAnalyzer_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local,
                }

                jobs.append(job)

    return jobs

def get_IOVs(jobConfig):
    return [[jobConfig['firstRun'], jobConfig['lastRun']]]
-- dummy change --
