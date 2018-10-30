from __future__ import print_function
import ConfigParser
import argparse
import pickle
import sys
import os
import subprocess
import shutil
import time


pickle_name = "dump.pkl" # contains all the measurement objects in a list
history_file = "history.log"
clock_interval = 20 # in seconds
STATE_NONE = -1
STATE_ITERATION_START=0
STATE_BJOBS_WAITING=1
STATE_BJOBS_DONE=2
STATE_BJOBS_FAILED=12
STATE_MERGE_WAITING=3
STATE_MERGE_DONE=4
STATE_MERGE_FAILED=14
STATE_SUMMARY_WAITING=5
STATE_SUMMARY_DONE=6
STATE_SUMMARY_FAILED=16
STATE_LOCAL_WAITING=7
STATE_LOCAL_DONE=8
STATE_LOCAL_FAILED=18
STATE_FINISHED=9

status_map = {}
status_map[STATE_NONE] = "STATE_NONE"
status_map[STATE_ITERATION_START] = "STATE_ITERATION_START"
status_map[STATE_BJOBS_WAITING] = "STATE_BJOBS_WAITING"
status_map[STATE_BJOBS_DONE] = "STATE_BJOBS_DONE"
status_map[STATE_BJOBS_FAILED] = "STATE_BJOBS_FAILED"
status_map[STATE_MERGE_WAITING] = "STATE_MERGE_WAITING"
status_map[STATE_MERGE_DONE] = "STATE_MERGE_DONE"
status_map[STATE_MERGE_FAILED] = "STATE_MERGE_FAILED"
status_map[STATE_SUMMARY_WAITING] = "STATE_SUMMARY_WAITING"
status_map[STATE_SUMMARY_DONE] = "STATE_SUMMARY_DONE"
status_map[STATE_SUMMARY_FAILED] = "STATE_SUMMARY_FAILED"
status_map[STATE_LOCAL_WAITING] = "STATE_LOCAL_WAITING"
status_map[STATE_LOCAL_DONE] = "STATE_LOCAL_DONE"
status_map[STATE_LOCAL_FAILED] = "STATE_LOCAL_FAILED"
status_map[STATE_FINISHED] = "STATE_FINISHED"
base = ""


def ensurePathExists(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def replaceAllRanges(string):
    if "[" in string and "]" in string:
        strings = []
        posS = string.find("[")
        posE = string.find("]")
        nums = string[posS+1:posE].split(",")
        expression = string[posS:posE+1]
        
        nums = string[string.find("[")+1:string.find("]")]
        for interval in nums.split(","):
            interval = interval.strip()
            if "-" in interval:
                lowNum = int(interval.split("-")[0])
                upNum = int(interval.split("-")[1])
                for i in range(lowNum, upNum+1):
                    newstring = string[0:posS]+str(i)+string[posE+1:]
                    newstring = replaceAllRanges(newstring)
                    strings += newstring
            else:
                newstring = string[0:posS]+interval+string[posE+1:]
                newstring = replaceAllRanges(newstring)
                strings += newstring
        return strings
    else:
        return [string,]



def save(measurements):
    with open(pickle_name, "w") as saveFile: 
        pickle.dump(measurements, saveFile)

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
            self.maxEvents = dsDict["maxEvents"]
        if dsDict.has_key("isMC"):
            if dsDict["isMC"] == "True":
                self.sampleType = "MC"
            else:
                self.sampleType ="data1"
                
        self.conditions = []
        for key, value in dsDict.items():
            if key.startswith("condition"):
                record = key.split(" ")[1]
                connect, tag = value.split(" ")
                self.conditions.append({"record":record, "connect":connect, "tag":tag})

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
        
        self.hasAlignmentCondition = False # If this is true, no other Alignment-Object is loaded in apeEstimator_cfg.py using the alignmentName
        self.conditions = []
        for key, value in alDict.items():
            if key.startswith("condition"):
                record = key.split(" ")[1]
                connect, tag = value.split(" ")
                if record == "TrackerAlignmentRcd":
                    self.hasAlignmentCondition = True
                self.conditions.append({"record":record, "connect":connect, "tag":tag})
        
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
    
    def __init__(self, name, dataset, alignment, additionalOptions={}):
        self.name = name
        self.alignment = alignment
        self.dataset = dataset
        self.curIteration = 0
        self.status = STATE_ITERATION_START
        self.runningJobs = []
        self.failedJobs = []
        self.startTime = subprocess.check_output(["date"]).strip()
        
        for key, value in additionalOptions.items():
            setattr(self, key, value)
            print(key, value)
        self.firstIteration=int(self.firstIteration)
        self.maxIterations=int(self.maxIterations)
        self.curIteration = self.firstIteration
        if self.alignment.isDesign:
            self.maxIterations = 0
            
        if self.alignment.isDesign and self.dataset.sampleType != "MC":
            # For now, this won't immediately shut down the program
            print("APE Measurement {} is scheduled to to an APE baseline measurement with a dataset that is not marked as isMC=True. Is this intended?".format(self.name))
        ensurePathExists('{}/hists/{}'.format(base, self.name))
        if not self.alignment.isDesign:
            ensurePathExists('{}/hists/{}/apeObjects'.format(base, self.name))
        
    def get_status(self):
        return status_map[self.status]
    
    def print_status(self):
        print("APE Measurement {} in iteration {} is now in status {}".format(self.name, self.curIteration, self.get_status()))
    
    def submit_jobs(self):
        toSubmit = []
        
        allConditions = self.alignment.conditions+self.dataset.conditions
        allConditions = list({v['record']:v for v in allConditions}.values()) # should we clean for duplicate records? the overlap record last defined (from dataset) 
                                                                              # will be kept in case of overlap, which is the same as if there was no overlap removal
        
        # If conditions are made, create file to load them from
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
        
        # create a batch job file for each input file
        for i in range(self.dataset.nFiles):
            inputFile = self.dataset.fileList[i]
            inputCommands = "sample={sample} fileNumber={fileNo} iterNumber={iterNo} lastIter={lastIter} alignRcd={alignRcd} maxEvents={maxEvents} globalTag={globalTag} measurementName={name} conditions={conditions}".format(sample=self.dataset.sampleType,fileNo=i+1,iterNo=self.curIteration,lastIter=lastIter,alignRcd=alignmentNameToUse, maxEvents=self.dataset.maxEvents, globalTag=self.alignment.globalTag, name=self.name, conditions=rawFileName)
            fiName = "{}/test/autoSubmitter/workingArea/batchscript_{}_iter{}_{}".format(base, self.name,self.curIteration,i+1)
            with open(fiName+".tcsh", "w") as jobFile:
                from autoSubmitterTemplates import bjobTemplate
                jobFile.write(bjobTemplate.format(inputFile = inputFile, inputCommands=inputCommands))
            toSubmit.append((fiName,i+1))
        
        # submit all batch jobs
        submitName = "{}/test/autoSubmitter/workingArea/submit_{}_jobs_iter{}.sh".format(base, self.name, self.curIteration)
        with open(submitName,"w") as submitFile:
            for sub,number in toSubmit:
                from autoSubmitterTemplates import submitJobTemplate
                errorFile = sub+"_error.txt"
                outputFile = sub+"_output.txt"
                jobFile = sub+".tcsh"
                date = subprocess.check_output(["date", "+%m_%d_%H_%M_%S"]).strip()
                jobName = sub.split("/")[-1]+"_"+date
                self.runningJobs.append((jobName, number))
                submitFile.write(submitJobTemplate.format(errorFile=errorFile, outputFile=outputFile, jobFile=jobFile, jobName=jobName))
            submitFile.write("rm -- $0")
        
        subOut = subprocess.check_output("bash {}".format(submitName), shell=True).strip()
        if len(subOut) == 0:
                print("Running on environment that does not know bsub command or ssh session is timed out (ongoing for longer than 24h?), exiting")
                sys.exit()
        self.status = STATE_BJOBS_WAITING
        self.print_status()
    
    def check_jobs(self):
        lastStatus = self.status
        stillRunningJobs = []
        # check all still running jobs
        for job, number in self.runningJobs:
            from autoSubmitterTemplates import checkJobTemplate
            checkString = checkJobTemplate.format(jobName=job)
            jobState = subprocess.check_output(checkString, shell=True).rstrip()
            if "DONE" in jobState:
                # Catch Exceptions that do not influence the job state but make the measurement fail anyway
                errFile = "{}/test/autoSubmitter/workingArea/batchscript_{}_iter{}_{}_error.txt".format(base, self.name,self.curIteration,number)
                foundErr = False
                with open(errFile, "r") as err:
                    for line in err:
                        if "Fatal Exception" in line.strip():
                            foundErr = True
                            break
                if foundErr:
                    print("Job {} in iteration {} of APE measurement {} has failed".format(job, self.curIteration, self.name))
                    self.failedJobs.append(job)               
                else:
                    print("Job {} in iteration {} of APE measurement {} has just finished".format(job, self.curIteration, self.name))
            elif "EXIT" in jobState:
                print("Job {} in iteration {} of APE measurement {} has failed".format(job, self.curIteration, self.name))
                self.failedJobs.append(job)
            elif "RUN" in jobState or "PEND" in jobState:
                stillRunningJobs.append((job, number))
            elif "Job <{}> is not found".format(job) in jobState:
                print("Job {} of APE measurement was not found in queue, so it is assumed that it successfully finished long ago.".format(job, self.name))
            elif len(jobState) == 0:
                print("Running on environment that does not know bjobs command or ssh session is timed out (ongoing for longer than 24h?), exiting")
                sys.exit()
            else:
                print("Unknown state {}, marking job {} of APE measurement {} as failed".format(jobState, job, self.name))
                self.failedJobs.append(job)
        self.runningJobs = stillRunningJobs
        
        # at least one job failed
        if len(self.failedJobs) > 0:
            self.status = STATE_BJOBS_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        elif len(self.runningJobs) == 0:
            self.status = STATE_BJOBS_DONE
            print("All batch jobs of APE measurement {} in iteration {} are done".format(self.name, self.curIteration))
        if lastStatus != self.status:
            self.print_status() 
        
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
        for job, number in self.runningJobs:
            subprocess.call(killJobTemplate.format(jobName=job), shell=True)
        self.runningJobs = []
        self.status = STATE_NONE
        
    def purge(self):
        self.kill()
        folderName = '{}/hists/{}'.format(base, self.name)
        shutil.rmtree(folderName)
        # remove log-files as well?
        
def main():    
    parser = argparse.ArgumentParser(description="Automatically run APE measurements")
    parser.add_argument("-c", "--config", action="append", dest="configs", default=[],
                          help="Config file that has list of measurements")
    parser.add_argument("-k", "--kill", action="append", dest="kill", default=[],
                          help="List of measurement names to kill (=remove from list and kill all bjobs)")
    parser.add_argument("-p", "--purge", action="append", dest="purge", default=[],
                          help="List of measurement names to purge (=kill and remove folder)")
    parser.add_argument("-r", "--resume", action="append", dest="resume", default=[],
                          help="Resume interrupted APE measurements which are stored in pickle files")
    parser.add_argument("-d", "--dump", action="append", dest="dump", default=[],
                          help='Specify in which .pkl file to store the measurements')

    args = parser.parse_args()
    
    global base
    global clock_interval
    global pickle_name
    
    if args.dump != []: # choose different file than default
        pickle_name = args.dump[0]
    
    try:
        base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
    except KeyError:
        print("No CMSSW environment was set, exiting")
        sys.exit()
    
    
    measurements = []
    
    if args.resume != []:
        for resumeFile in args.resume:
            try:
                with open(resumeFile, "r") as saveFile: 
                    resumed = pickle.load(saveFile)
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
            except IOError:
                print("Could not resume because {} could not be opened, exiting".format(pickle_name))
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
                key = setting.split("=")[0]
                value = setting.split("=")[1]
                additionalOptions[key] = value
                
            measurement = ApeMeasurement(name, dataset, alignment, additionalOptions)
            
            measurements.append(measurement)
            
            print("APE Measurement {} was started".format(measurement.name))
            
    
    while True:
        measurements = [measurement for measurement in measurements if not measurement.status==STATE_NONE]
        save(measurements)
        
        if len(measurements) == 0:
            print("No APE measurements are active, exiting")
            break
        for measurement in measurements:
            if measurement.status == STATE_ITERATION_START:
                # start bjobs
                print("APE Measurement {} just started iteration {}".format(measurement.name, measurement.curIteration))
                measurement.submit_jobs()
                save(measurements)
                continue # no reason to immediately check jobs
            if measurement.status == STATE_BJOBS_WAITING:
                # check if bjobs are finished
                measurement.check_jobs()
                save(measurements)
                #~ if measurement.status == STATE_BJOBS_DONE:
                    #~ continue # give time for files to be closed and delivered back from the batch jobs
            if measurement.status == STATE_BJOBS_DONE:
                # merge files
                measurement.do_merge()
                save(measurements)
            if measurement.status == STATE_MERGE_DONE:
                # start summary
                measurement.do_summary()
                save(measurements)
            if measurement.status == STATE_SUMMARY_DONE:
                # start local setting (only if not a baseline measurement)
                if measurement.alignment.isDesign:
                    measurement.status = STATE_LOCAL_DONE
                else:
                    measurement.do_local_setting()
                save(measurements)
            if measurement.status == STATE_LOCAL_DONE:
                measurement.finish_iteration()
                save(measurements)
                # go to next iteration or finish measurement
            if measurement.status == STATE_BJOBS_FAILED or \
                measurement.status == STATE_MERGE_FAILED or \
                measurement.status == STATE_SUMMARY_FAILED or \
                measurement.status == STATE_LOCAL_FAILED or \
                measurement.status == STATE_FINISHED:
                    # might want to do something different. for now, this is the solution
                    with open(history_file, "a") as fi:
                        fi.write("APE measurement {name} which was started at {start} finished at {end} with state {state} in iteration {iteration}\n".format(name=measurement.name, start=measurement.startTime, end=measurement.finishTime, state=measurement.get_status(), iteration=measurement.curIteration))
                    measurement.status = STATE_NONE
                    save(measurements)
            if measurement.status == STATE_ITERATION_START: # this ensures that jobs do not go into idle if many measurements are done simultaneously
                # start bjobs
                print("APE Measurement {} just started iteration {}".format(measurement.name, measurement.curIteration))
                measurement.submit_jobs()
                save(measurements)           
        
        
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
            
if __name__ == "__main__":
    main()
