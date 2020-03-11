from __future__ import print_function
import configparser as ConfigParser
import argparse
import shelve
import sys
import os
import subprocess
import threading
import shutil
import time
import re
import ROOT
from helpers import *
sys.path.append("../plottingTools")

shelve_name = "dump.shelve" # contains all the measurement objects and plot objects
history_file = "history.log"
clock_interval = 20 # in seconds
delete_logs_after_finish = False  # if it is not desired to keep the log and submit script files
use_caf = False

def save(name, object):
    # in case of multiple threads running this stops potentially problematic file access
    global lock
    lock.acquire() 
    try:
        sh = shelve.open(shelve_name)
        sh[name] = object
        sh.close()
    finally:
        lock.release()

class Dataset:
    name = ""
    nFiles = 0
    maxEvents = -1
    baseDirectory = ""
    sampleType = "data1"
    fileList = []
    conditions = []
    
    def __init__(self, config, name):
        dsDict = dict(config.items("dataset:{}".format(name)))
        self.name = name
        self.baseDirectory = dsDict["baseDirectory"]
        
        self.fileList = []
        names = dsDict["fileNames"].split(" ")
        for name in names:
            parsedNames = replaceAllRanges(name)
            for fileName in parsedNames:
                self.fileList.append(self.baseDirectory+"/"+fileName)
        self.nFiles = len(self.fileList)

        if dsDict.has_key("maxEvents"):
            self.maxEvents = int(dsDict["maxEvents"])
        if dsDict.has_key("isMC"):
            if dsDict["isMC"] == "True":
                self.sampleType = "MC"
            else:
                self.sampleType ="data1"
        
        self.conditions, dummy, self.validConditions = loadConditions(dsDict)      
        
        # check if any of the sources used for conditions is invalid
        if not self.validConditions:
            print("Invalid conditions defined for dataset {}".format(self.name))
        

class Alignment:        
    name = ""
    alignmentName = None
    baselineDir = "Design"
    globalTag = "None"
    isDesign = False
    hasAlignmentCondition = False
    conditions = []
    
    def __init__(self, config, name):
        alDict = dict(config.items("alignment:{}".format(name)))
        self.name = name
        if alDict.has_key("alignmentName"):
            self.alignmentName = alDict["alignmentName"]
        if alDict.has_key("globalTag"):
            self.globalTag = alDict["globalTag"]
        if alDict.has_key("baselineDir"):
            self.baselineDir= alDict["baselineDir"]
        if alDict.has_key("isDesign"):
            self.isDesign= (alDict["isDesign"] == "True")
        
        # If self.hasAlignmentCondition is true, no other Alignment-Object is loaded in apeEstimator_cfg.py using the alignmentName
        self.conditions, self.hasAlignmentCondition, self.validConditions = loadConditions(alDict) 
        
        # check if any of the sources used for conditions is invalid
        if not self.validConditions:
            print("Invalid conditions defined for alignment {}".format(self.name))
        
        
        # check if at least one of the two ways to define the alignment was used
        if self.alignmentName == None and not self.hasAlignmentCondition:
            print("Error: No alignment object name or record was defined for alignment {}".format(self.name))
            sys.exit()
        

class ApeMeasurement:
    name = "workingArea"
    curIteration = 0
    firstIteration = 0
    maxIterations = 15
    status = STATE_NONE
    dataset = None
    alignment = None
    runningJobs  = None
    failedJobs      = None
    startTime = ""
    finishTime = ""
    
    def __init__(self, name, dataset, alignment, config, additionalOptions={}):
        self.name = name
        self.alignment = alignment
        self.dataset = dataset
        self.curIteration = 0
        self.status = STATE_ITERATION_START
        self.runningJobs = []
        self.failedJobs = []
        self.startTime = subprocess.check_output(["date"]).strip()
        
        self.maxEvents = self.dataset.maxEvents
        # standards for result plot
        self.resultPlotTitle=""
        self.resultPlotLabel=self.name
        self.resultPlotDo=False
        self.resultPlotOutPath = '{}/hists/{}/'.format(base, self.name)
        
        for key, value in additionalOptions.items():
            setattr(self, key, value)
        self.firstIteration=int(self.firstIteration)
        self.maxIterations=int(self.maxIterations)
        self.curIteration = self.firstIteration
        self.maxEvents = int(self.maxEvents)
        if self.alignment.isDesign:
            self.maxIterations = 0
        
        
        if not self.alignment.validConditions or not self.dataset.validConditions:
            self.status = STATE_INVALID_CONDITIONS
            self.print_status()
            self.finishTime = subprocess.check_output(["date"]).strip()
            return
        
            
        if self.alignment.isDesign and self.dataset.sampleType != "MC":
            # For now, this won't immediately shut down the program
            print("APE Measurement {} is scheduled to to an APE baseline measurement with a dataset that is not marked as isMC=True. Is this intended?".format(self.name))
        ensurePathExists('{}/hists/{}'.format(base, self.name))
        if not self.alignment.isDesign:
            ensurePathExists('{}/hists/{}/apeObjects'.format(base, self.name))
        
        
        if self.resultPlotDo == "True":
            self.resultPlotTitle = self.resultPlotTitle.replace("~", " ")
            # Adds new section to config file, result plots are loaded from the config in a later step
            sectionName = "resultplot:{}".format(self.name)
            config.add_section(sectionName)
            config.set(sectionName, "wait {}".format(self.resultPlotLabel), "{} iteration={}".format(self.name, min(14, self.maxIterations-1)))
            config.set(sectionName, "title", self.resultPlotTitle)
            config.set(sectionName, "outPath", self.resultPlotOutPath)
            
    def get_status(self):
        return status_map[self.status]
    
    def print_status(self):
        print("APE Measurement {} in iteration {} is now in status {}".format(self.name, self.curIteration, self.get_status()))
    
    # submit jobs for track refit and hit categorization
    def submit_jobs(self):
        toSubmit = []
        
        allConditions = self.alignment.conditions+self.dataset.conditions
        allConditions = list({v['record']:v for v in allConditions}.values()) # should we clean for duplicate records? the record last defined (from dataset) 
                                                                              # will be kept in case of overlap, which is the same as if there was no overlap removal
        
        # If conditions are made, create file to load them from
        rawFileName = "None"
        conditionsFileName = "None"
        if len(allConditions) > 0:
            conditionsFileName = "{base}/python/conditions/conditions_{name}_iter{iterNo}_cff.py".format(base=base,name=self.name, iterNo=self.curIteration)
            rawFileName = "conditions_{name}_iter{iterNo}_cff".format(name=self.name, iterNo=self.curIteration)
            with open(conditionsFileName, "w") as fi:
                from autoSubmitterTemplates import conditionsFileHeader
                fi.write(conditionsFileHeader)
                from autoSubmitterTemplates import conditionsTemplate
                for condition in allConditions:
                    fi.write(conditionsTemplate.format(record=condition["record"], connect=condition["connect"], tag=condition["tag"]))
                
                
        alignmentNameToUse = self.alignment.alignmentName
        if self.alignment.hasAlignmentCondition:
                alignmentNameToUse = "fromConditions"
        
        lastIter = (self.curIteration==self.maxIterations) and not self.alignment.isDesign
        
        inputCommands = "sample={sample} fileNumber={fileNo} iterNumber={iterNo} lastIter={lastIter} alignRcd={alignRcd} maxEvents={maxEvents} globalTag={globalTag} measurementName={name} conditions={conditions}".format(sample=self.dataset.sampleType,fileNo="$1",iterNo=self.curIteration,lastIter=lastIter,alignRcd=alignmentNameToUse, maxEvents=self.maxEvents, globalTag=self.alignment.globalTag, name=self.name, conditions=rawFileName)
        
        from autoSubmitterTemplates import condorJobTemplate
        jobFileContent = condorJobTemplate.format(base=base, inputFile="$2", inputCommands=inputCommands)
        jobFileName = "{}/test/autoSubmitter/workingArea/batchscript_{}_iter{}.tcsh".format(base, self.name,self.curIteration)
        with open(jobFileName, "w") as jobFile:
            jobFile.write(jobFileContent)
        
        # create a batch job file for each input file
        arguments = ""
        from autoSubmitterTemplates import condorArgumentTemplate
        for i in range(self.dataset.nFiles):
            inputFile = self.dataset.fileList[i]
            fileNumber = i+1
            arguments += condorArgumentTemplate.format(fileNumber=fileNumber, inputFile=inputFile)
            
        # build condor submit script
        date = subprocess.check_output(["date", "+%m_%d_%H_%M_%S"]).strip()
        sub = "{}/test/autoSubmitter/workingArea/job_{}_iter{}".format(base, self.name, self.curIteration)
        
        errorFileTemp  = sub+"_error_{}.txt"
        errorFile  = errorFileTemp.format("$(ProcId)")
        outputFile = sub+"_output_$(ProcId).txt"
        logFileTemp= sub+"_condor_{}.log"
        logFile    = logFileTemp.format("$(ProcId)")
        jobFile    = sub+".tcsh"   
        jobName    = "{}_{}".format(self.name, self.curIteration)
        for i in range(self.dataset.nFiles):
            # make file empty if it existed before
            with open(logFileTemp.format(i), "w") as fi:
                pass
        
        # create submit file
        from autoSubmitterTemplates import condorSubTemplate
        from autoSubmitterTemplates import condorSubTemplateCAF
        if use_caf:
            submitFileContent = condorSubTemplateCAF.format(jobFile=jobFileName, outputFile=outputFile, errorFile=errorFile, logFile=logFile, arguments=arguments, jobName=jobName)
        else:
            submitFileContent = condorSubTemplate.format(jobFile=jobFileName, outputFile=outputFile, errorFile=errorFile, logFile=logFile, arguments=arguments, jobName=jobName)
        submitFileName = "{}/test/autoSubmitter/workingArea/submit_{}_jobs_iter{}.sub".format(base, self.name, self.curIteration)
        with open(submitFileName, "w") as submitFile:
            submitFile.write(submitFileContent)
        
        # submit batch
        from autoSubmitterTemplates import submitCondorTemplate
        subOut = subprocess.check_output(submitCondorTemplate.format(subFile=submitFileName), shell=True).strip()
    
        if len(subOut) == 0:
                print("Running on environment that does not know bsub command or ssh session is timed out (ongoing for longer than 24h?), exiting")
                sys.exit()
                
        cluster = subOut.split(" ")[-1][:-1]
        for i in range(self.dataset.nFiles):
            # list contains condor log files from which to read when job is terminated to detect errors
            self.runningJobs.append((logFileTemp.format(i), errorFileTemp.format(i), "{}.{}".format(cluster, i)))
        
        
        self.status = STATE_BJOBS_WAITING
        self.print_status()
    
    def check_jobs(self):
        lastStatus = self.status
        stillRunningJobs = []
        # check all still running jobs
        for logName, errName, jobId in self.runningJobs:
            # read condor logs instead of doing condor_q or similar, as it is much faster
            if not os.path.isfile(logName):
                print("{} does not exist even though it should, marking job as failed".format(logName))
                self.failedJobs.append( (logName, errName) ) 
                break
            with open(logName, "r") as logFile:
                log = logFile.read()
            if not "submitted" in log:
                print("{} was apparently not submitted, did you empty the log file or is condor not working?".format(jobId))
                self.failedJobs.append( (logName, errName) ) 
                
            if "Job was aborted" in log:
                print("Job {} of measurement {} in iteration {} was aborted".format(jobId, self.name, self.curIteration))
                self.failedJobs.append( (logName, errName) ) 
            elif "Job terminated" in log:
                if "Normal termination (return value 0)" in log:
                    foundErr = False
                    with open(errName, "r") as err:
                        for line in err:
                            if "Fatal Exception" in line.strip():
                                foundErr = True
                                break
                    if not foundErr:
                        print("Job {} of measurement {} in iteration {} finished successfully".format(jobId, self.name, self.curIteration))
                    else:
                        # Fatal error in stderr
                        print("Job {} of measurement {} in iteration {} has a fatal error, check stderr".format(jobId, self.name, self.curIteration))
                        self.failedJobs.append( (logName, errName) ) 
                else:
                    # nonzero return value
                    print("Job {} of measurement {} in iteration {} failed, check stderr".format(jobId, self.name, self.curIteration))
                    self.failedJobs.append( (logName, errName) ) 
            else:
                stillRunningJobs.append( (logName, errName, jobId) )
        self.runningJobs = stillRunningJobs
        
        # at least one job failed
        if len(self.failedJobs) > 0:
            self.status = STATE_BJOBS_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        elif len(self.runningJobs) == 0:
            self.status = STATE_BJOBS_DONE
            print("All condor jobs of APE measurement {} in iteration {} are done".format(self.name, self.curIteration))
            
            # remove files
            if delete_logs_after_finish:
                submitFile = "{}/test/autoSubmitter/workingArea/submit_{}_jobs_iter{}.sub".format(base, self.name, self.curIteration)
                jobFile = "{}/test/autoSubmitter/workingArea/batchscript_{}_iter{}.tcsh".format(base, self.name,self.curIteration)
                os.remove(submitFile)
                os.remove(jobFile)
            
                for i in range(self.dataset.nFiles):
                    sub = "{}/test/autoSubmitter/workingArea/job_{}_iter{}".format(base, self.name, self.curIteration)
                    errorFile  = sub+"_error_{}.txt".format(i)
                    outputFile = sub+"_output_{}.txt".format(i)
                    logFile    = sub+"_condor_{}.log".format(i) 
                    os.remove(errorFile)
                    os.remove(outputFile)
                    os.remove(logFile)
                    
        if lastStatus != self.status:
            self.print_status() 
    
    # merges files from jobs
    def do_merge(self):
        self.status = STATE_MERGE_WAITING
        if self.alignment.isDesign:
            folderName = '{}/hists/{}/baseline'.format(base, self.name)
        else:
            folderName = '{}/hists/{}/iter{}'.format(base, self.name, self.curIteration)
        
        # (re)move results from previous measurements before creating folder
        if os.path.isdir(folderName):
            if os.path.isdir(folderName+"_old"):
                shutil.rmtree("{}_old".format(folderName))
            os.rename(folderName, folderName+"_old")
        os.makedirs(folderName)
        
        # This is so that the structure of the tree can be retrieved by ApeEstimatorSummary.cc and the tree does not have to be rebuilt
        if self.curIteration > 0 and not self.alignment.isDesign: # don't have to check for isDesign here because it always ends after iteration 0...
            shutil.copyfile('{}/hists/{}/iter{}/allData_iterationApe.root'.format(base, self.name, self.curIteration-1),folderName+"/allData_iterationApe.root")
        fileNames = ['{}/hists/{}/{}{}.root'.format(base, self.name, self.dataset.sampleType, str(i)) for i in range(1, self.dataset.nFiles+1)]
        fileString = " ".join(fileNames)
        
        from autoSubmitterTemplates import mergeTemplate
        merge_result = subprocess.call(mergeTemplate.format(path=folderName, inputFiles=fileString), shell=True) # returns exit code (0 if no error occured)
        for name in fileNames:
            os.remove(name)
            
        if os.path.isfile("{}/allData.root".format(folderName)) and merge_result == 0: # maybe check with ROOT if all neccessary contents are in?
            self.status = STATE_MERGE_DONE
        else:
            self.status = STATE_MERGE_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        self.print_status()
    
    # calculates APE
    def do_summary(self):
        self.status = STATE_SUMMARY_WAITING        
        from autoSubmitterTemplates import summaryTemplate
        if self.alignment.isDesign:
            #use measurement name as baseline folder name in this case
            inputCommands = "iterNumber={} setBaseline={} measurementName={} baselineName={}".format(self.curIteration,self.alignment.isDesign,self.name, self.name)
        else:
            inputCommands = "iterNumber={} setBaseline={} measurementName={} baselineName={}".format(self.curIteration,self.alignment.isDesign,self.name, self.alignment.baselineDir)
        
        summary_result = subprocess.call(summaryTemplate.format(inputCommands=inputCommands), shell=True) # returns exit code (0 if no error occured)
        if summary_result == 0:
            self.status = STATE_SUMMARY_DONE
        else:
            self.status = STATE_SUMMARY_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        self.print_status()
    
    # saves APE to .db file so it can be read out next iteration
    def do_local_setting(self):
        self.status = STATE_LOCAL_WAITING       
        from autoSubmitterTemplates import localSettingTemplate
        inputCommands = "iterNumber={} setBaseline={} measurementName={}".format(self.curIteration,self.alignment.isDesign,self.name)

        local_setting_result = subprocess.call(localSettingTemplate.format(inputCommands=inputCommands), shell=True) # returns exit code (0 if no error occured)
        if local_setting_result == 0:
            self.status = STATE_LOCAL_DONE
        else:
            self.status = STATE_LOCAL_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        self.print_status()
        
    def finish_iteration(self):
        print("APE Measurement {} just finished iteration {}".format(self.name, self.curIteration))
        if self.curIteration < self.maxIterations:
            self.curIteration += 1
            self.status = STATE_ITERATION_START
        else:
            self.status = STATE_FINISHED
            self.finishTime = subprocess.check_output(["date"]).strip()
            print("APE Measurement {}, which was started at {} was finished after {} iterations, at {}".format(self.name, self.startTime, self.curIteration, self.finishTime))
            
    def kill(self):
        from autoSubmitterTemplates import killJobTemplate
        for log, err, jobId in self.runningJobs:
            subprocess.call(killJobTemplate.format(jobId=jobId), shell=True)
        self.runningJobs = []
        self.status = STATE_NONE
        
    def purge(self):
        self.kill()
        folderName = '{}/hists/{}'.format(base, self.name)
        shutil.rmtree(folderName)
        # remove log-files as well?
        
    def run_iteration(self):
        global threadcounter
        global measurements
        threadcounter.acquire()
        try:
            if self.status == STATE_ITERATION_START:
                # start bjobs
                print("APE Measurement {} just started iteration {}".format(self.name, self.curIteration))
                while True: 
                    try:
                        self.submit_jobs()
                        save("measurements", measurements)
                        break
                    except KeyboardInterrupt:
                        exit()
                    except:
                        # this is needed in case the scheduler goes down
                        print("Error submitting, waiting for 1 minute")
                        time.sleep(60)
                    
                    continue # no reason to immediately check jobs
            if self.status == STATE_BJOBS_WAITING:
                # check if bjobs are finished
                self.check_jobs()
                save("measurements", measurements)
            if self.status == STATE_BJOBS_DONE:
                # merge files
                self.do_merge()
                save("measurements", measurements)
            if self.status == STATE_MERGE_DONE:
                # start summary
                self.do_summary()
                save("measurements", measurements)
            if self.status == STATE_SUMMARY_DONE:
                # start local setting (only if not a baseline measurement)
                if self.alignment.isDesign:
                    self.status = STATE_LOCAL_DONE
                else:
                    self.do_local_setting()
                save("measurements", measurements)
            if self.status == STATE_LOCAL_DONE:
                self.finish_iteration()
                save("measurements", measurements)
                # go to next iteration or finish measurement
            if self.status == STATE_BJOBS_FAILED or \
                self.status == STATE_MERGE_FAILED or \
                self.status == STATE_SUMMARY_FAILED or \
                self.status == STATE_LOCAL_FAILED or \
                self.status == STATE_INVALID_CONDITIONS or \
                self.status == STATE_FINISHED:
                    with open(history_file, "a") as fi:
                        fi.write("APE measurement {name} which was started at {start} finished at {end} with state {state} in iteration {iteration}\n".format(name=self.name, start=self.startTime, end=self.finishTime, state=self.get_status(), iteration=self.curIteration))
                    if self.status == STATE_FINISHED:
                        global finished_measurements
                        finished_measurements[self.name] = self
                        save("finished", finished_measurements)
                    else:
                        global failed_measurements
                        failed_measurements[self.name] = self
                        self.status = STATE_NONE
                        save("failed", failed_measurements)
                    save("measurements", measurements)
            if self.status == STATE_ITERATION_START: # this ensures that jobs do not go into idle if many measurements are done simultaneously
                # start bjobs
                print("APE Measurement {} just started iteration {}".format(self.name, self.curIteration))
                self.submit_jobs()
                save("measurements", measurements)   
        finally:
            threadcounter.release()
        
class ResultPlot:
    def __init__(self, config, name):
        rpDict = dict(config.items(name))
        self.waitingFor = []
        self.loadingFrom = []
        self.making = []
        self.name = name.split("resultplot:")[1]
        self.outPath = "{}/hists/{}/".format(base,self.name)
        self.title = ""
        self.granularity = "standardGranularity"
        
        for key, value in rpDict.items():
            
            if key.startswith("wait ") or key.startswith("load "):
                optsList = value.split(" ")
                val = optsList[0]
                optsDict = {}
                for i in range(1, len(optsList)):
                    k, v = optsList[i].split("=")
                    if k == "marker" or k == "color":
                        v = eval(v) # convert to int, makes it possible to use ROOT variables
                    optsDict[k] = v
                
            
            if key.startswith("wait "):
                label = key.split(" ")[1] 
                label = label.replace("~", " ")
                
                if optsDict.has_key("iteration"):
                    iteration = optsDict["iteration"]
                    optsDict.pop("iteration")
                else:
                    iteration = "14"
                
                self.waitingFor.append((val, iteration, label, optsDict.copy()))
            elif key.startswith("load "):
                label = key.split(" ")[1] 
                label = label.replace("~", " ")
                self.loadingFrom.append((val,label, optsDict.copy()))
            else:
                setattr(self, key, value)
        
    def check_finished(self, finished_measurements):
        for waiting in self.waitingFor:
            if not waiting[0] in finished_measurements.keys():
                return False
        return True
        
    def do_plot(self):
        import sys
        from resultPlotter import ResultPlotter
        import granularity

        plotter = ResultPlotter()
        plotter.setOutputPath(self.outPath)
        plotter.setTitle(self.title)
        plotter.setGranularity(getattr(granularity, self.granularity))
        for path, label, optsDict in self.loadingFrom:
            plotter.addInputFile(label, path, **optsDict)
        for name, iteration, label, optsDict in self.waitingFor:
            path = '{}/hists/{}/iter{}/allData_iterationApe.root'.format(base, name, iteration)
            plotter.addInputFile(label, path, **optsDict)
        ensurePathExists(self.outPath)
        plotter.draw()


def main():    
    parser = argparse.ArgumentParser(description="Automatically run APE measurements")
    parser.add_argument("-c", "--config", action="append", dest="configs", default=[],
                          help="Config file that has list of measurements")
    parser.add_argument("-k", "--kill", action="append", dest="kill", default=[],
                          help="List of measurement names to kill (=remove from list and kill all bjobs)")
    parser.add_argument("-p", "--purge", action="append", dest="purge", default=[],
                          help="List of measurement names to purge (=kill and remove folder)")
    parser.add_argument("-r", "--resume", action="append", dest="resume", default=[],
                          help="Resume interrupted APE measurements which are stored in shelves (specify shelves)")
    parser.add_argument("-d", "--dump", action="store", dest="dump", default=None,
                          help='Specify in which .shelve file to store the measurements and plots')
    parser.add_argument("-n", "--ncores", action="store", dest="ncores", default=1, type=int,
                          help='Number of threads running in parallel')
    parser.add_argument("-C", "--caf",action="store_true", dest="caf", default=False,
                                              help="Use CAF queue for condor jobs")
    args = parser.parse_args()
    
    global base
    global clock_interval
    global shelve_name
    global threadcounter
    global lock
    global use_caf
    
    use_caf = args.caf
    enableCAF(use_caf)
    
    
    threadcounter = threading.BoundedSemaphore(args.ncores)
    lock = threading.Lock()
    
    if args.dump != None: # choose different file than default
        shelve_name = args.dump
    elif args.resume != []:
        shelve_name = args.resume[0]
    try:
        base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
    except KeyError:
        print("No CMSSW environment was set, exiting")
        sys.exit()
    
    killTargets = []
    purgeTargets = []
    for toConvert in args.kill:
        killTargets += replaceAllRanges(toConvert)
        
    for toConvert in args.purge:
        purgeTargets += replaceAllRanges(toConvert)
    
    global measurements
    measurements = []
    global finished_measurements
    finished_measurements = {}
    global failed_measurements
    failed_measurements = {}
    resultPlots = []
    
    if args.resume != []:
        for resumeFile in args.resume:
            try:
                sh = shelve.open(resumeFile)
                resumed = sh["measurements"]
                resumed_plots = sh["resultPlots"]
                
                resumed_failed = sh["failed"]
                resumed_finished = sh["finished"]
                sh.close()
                
                for res in resumed:
                    measurements.append(res)
                    print("Measurement {} in state {} in iteration {} was resumed".format(res.name, res.get_status(), res.curIteration))
                    # Killing and purging is done here, because it doesn't make 
                    # sense to kill or purge a measurement that was just started
                    for to_kill in args.kill:
                        if res.name == to_kill:
                            res.kill()
                    for to_purge in args.purge:
                        if res.name == to_purge:
                            res.purge()
                for res in resumed_plots:
                    resultPlots.append(res)
                    print("Result plot {} was resumed".format(res.name))
                
                failed_measurements.update(resumed_failed)
                finished_measurements.update(resumed_finished)
                
            except IOError:
                print("Could not resume because {} could not be opened, exiting".format(shelve_name))
                sys.exit()
            
    # read out from config file
    if args.configs != []:
        config = ConfigParser.RawConfigParser()
        config.optionxform = str 
        config.read(args.configs)
    
        for name, opts in config.items("measurements"):
            if name in map(lambda x: x.name ,measurements):
                print("Error: APE Measurement with name {} already exists, skipping".format(name))
                continue
            
            settings = opts.split(" ")
            if len(settings) < 2:
                print("Error: number of arguments for APE Measurement {} is insufficient".format(name))
                sys.exit()
                
            datasetID = settings[0].strip()
            alignmentID = settings[1].strip()

            dataset = Dataset(config, datasetID)
            alignment = Alignment(config, alignmentID)
            additionalOptions = {}
            
            for i in range(2, len(settings)):
                setting = settings[i].strip()
                key, value = setting.split("=")
                additionalOptions[key] = value
                
            measurement = ApeMeasurement(name, dataset, alignment, config, additionalOptions)
            
            measurements.append(measurement)
            
            print("APE Measurement {} was started".format(measurement.name))
        
        for name in config.sections():
            if name.startswith("resultplot:"):
                if not name.split("resultplot:")[1] in map(lambda x: x.name ,resultPlots):
                    resultPlots.append(ResultPlot(config, name))
                    print("Result plot {} was queued".format(name))
    
    while True:
        # remove finished and failed measurements
        measurements = [measurement for measurement in measurements if not (measurement.status==STATE_NONE or measurement.status == STATE_FINISHED)]
        save("measurements", measurements)
        save("resultPlots", resultPlots)
        save("failed", failed_measurements)
        save("finished", finished_measurements)
        
        list_threads = []
        for measurement in measurements:
            t = threading.Thread(target=measurement.run_iteration)
            list_threads.append(t)
            t.start()
        
        # wait for iterations to finish
        for t in list_threads:
            t.join()
     
        
        # Check if there are plots to do
        changed = False
        tempList = []
        for plot in resultPlots:
            if plot.check_finished(finished_measurements):
                plot.do_plot()
                changed = True
                with open(history_file, "a") as fi:
                        fi.write("Result plot {name} was created in folder {outPath}\n".format(name=plot.name, outPath=plot.outPath))
            else:
                tempList.append(plot)
        resultPlots = tempList
        tempList = None
        if changed:
            save("resultPlots", resultPlots)
        
        
        if len(measurements) == 0:
            print("No APE measurements are active, exiting")
            break
        
        
        try: # so that interrupting does not give an error message and just ends the program
            time_remaining = clock_interval
            while time_remaining > 0:
                print("Sleeping for {} seconds, you can safely [CTRL+C] now".format(time_remaining))
                time.sleep(1)
                time_remaining -= 1
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
            print("")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        except KeyboardInterrupt:
            sys.exit(0)
        
            
if __name__ == "__main__":
    main()
