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
status_map = {}
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
    def __init__(self, config, name):
        dsDict = dict(config.items("dataset:%s"%(name)))
        self.name = name
        self.baseDirectory = dsDict["baseDirectory"]
        
        self.fileList = []
        names = dsDict["fileNames"].split(" ")
        for name in names:
            if "[" in name and "]" in name:
                posS = name.find("[")
                posE = name.find("]")
                nums = name[posS+1:posE].split(",")
                expression = name[posS:posE+1]
                for interval in nums:
                    interval = interval.strip()
                    if "-" in interval:
                        lowNum = int(interval.split("-")[0])
                        upNum = int(interval.split("-")[1])
                        for i in range(lowNum, upNum+1):
                            self.fileList.append("%s/%s"%(self.baseDirectory, name.replace(expression, str(i))))
                    else:
                        self.fileList.append("%s/%s"%(self.baseDirectory, name.replace(expression, interval)))
            else:
                self.fileList.append("%s/%s"%(self.baseDirectory, name))
        self.nFiles = len(self.fileList)
        
        if dsDict.has_key("maxEvents"):
            self.maxEvents = dsDict["maxEvents"]
        if dsDict.has_key("isMC"):
            if dsDict["isMC"] == "True":
                self.sampleType = "MC"
            else:
                self.sampleType ="data1"

class Alignment:
    name = ""
    alignmentName = ""
    baselineDir = "Design"
    globalTag = "None"
    isDesign = False
    
    def __init__(self, config, name):
        alDict = dict(config.items("alignment:%s"%(name)))
        self.name = name
        self.alignmentName = alDict["alignmentName"]
        if alDict.has_key("globalTag"):
            self.globalTag = alDict["globalTag"]
        if alDict.has_key("baselineDir"):
            self.baselineDir= alDict["baselineDir"]
        if alDict.has_key("isDesign"):
            self.isDesign= (alDict["isDesign"] == "True")
            
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
            print("APE Measurement %s is scheduled to to an APE baseline measurement with a dataset that is not marked as isMC=True. Is this intended?"%(self.name))
        ensurePathExists('%s/hists/%s'%(base, self.name))
        if not self.alignment.isDesign:
            ensurePathExists('%s/hists/%s/apeObjects'%(base, self.name))
        
    def get_status(self):
        return status_map[self.status]
    
    def print_status(self):
        print("APE Measurement %s in iteration %d is now in status %s"%(self.name, self.curIteration, self.get_status()))
    
    def submit_jobs(self):
        toSubmit = []
        for i in range(self.dataset.nFiles):
            inputFile = self.dataset.fileList[i]
            lastIter = (self.curIteration==self.maxIterations) and not self.alignment.isDesign
            inputCommands = "sample=%s fileNumber=%s iterNumber=%d lastIter=%r alignRcd=%s maxEvents=%s globalTag=%s measurementName=%s"%(self.dataset.sampleType,i+1,self.curIteration,lastIter,self.alignment.alignmentName, self.dataset.maxEvents, self.alignment.globalTag, self.name)
            fiName = "%s/test/autoSubmitter/workingArea/batchscript_%s_iter%d_%d"%(base, self.name,self.curIteration,i+1)
            with open(fiName+".tcsh", "w") as jobFile:
                from autoSubmitterTemplates import bjobTemplate
                jobFile.write(bjobTemplate.format(inputFile = inputFile, inputCommands=inputCommands))
            toSubmit.append((fiName,i+1))
            
        submitName = "%s/test/autoSubmitter/workingArea/submit_%s_jobs_iter%d.sh"%(base, self.name, self.curIteration)
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
        
        subOut = subprocess.check_output("bash %s"%(submitName), shell=True).strip()
        if len(subOut) == 0:
                print("Running on environment that does not know bsub command or ssh session is timed out (ongoing for longer than 24h?), exiting")
                sys.exit()
        self.status = STATE_BJOBS_WAITING
        self.print_status()
    
    def check_jobs(self):
        lastStatus = self.status
        stillRunningJobs = []
        for job, number in self.runningJobs:
            from autoSubmitterTemplates import checkJobTemplate
            checkString = checkJobTemplate.format(jobName=job)
            jobState = subprocess.check_output(checkString, shell=True).rstrip()
            if "DONE" in jobState:
                # Catch Exceptions that do not influence the job state
                errFile = "%s/test/autoSubmitter/workingArea/batchscript_%s_iter%d_%d_error.txt"%(base, self.name,self.curIteration,number)
                foundErr = False
                with open(errFile, "r") as err:
                    for line in err:
                        if "Fatal Exception" in line.strip():
                            foundErr = True
                            break
                if foundErr:
                    print("Job %s in iteration %d of APE measurement %s has failed"%(job, self.curIteration, self.name))
                    self.failedJobs.append(job)               
                else:
                    print("Job %s in iteration %d of APE measurement %s has just finished"%(job, self.curIteration, self.name))
            elif "EXIT" in jobState:
                print("Job %s in iteration %d of APE measurement %s has failed"%(job, self.curIteration, self.name))
                self.failedJobs.append(job)
            elif "RUN" in jobState or "PEND" in jobState:
                stillRunningJobs.append((job, number))
            elif "Job <%s> is not found" in jobState:
                print("Job %s of APE measurement was not found in queue, so it is assumed that it successfully finished long ago."%(job, self.name))
            elif len(jobState) == 0:
                print("Running on environment that does not know bjobs command or ssh session is timed out (ongoing for longer than 24h?), exiting")
                sys.exit()
            else:
                print("Unknown state %s, marking job %s of APE measurement %s as failed"%(jobState, job, self.name))
                self.failedJobs.append(job)
        self.runningJobs = stillRunningJobs
        
        if len(self.failedJobs) > 0:
            self.status = STATE_BJOBS_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        elif len(self.runningJobs) == 0:
            self.status = STATE_BJOBS_DONE
            print("All batch jobs of APE measurement %s in iteration %d are done"%(self.name, self.curIteration))
        if lastStatus != self.status:
            self.print_status() 
        
    def do_merge(self):
        self.status = STATE_MERGE_WAITING
        if self.alignment.isDesign:
            folderName = '%s/hists/%s/baseline'%(base, self.name)
        else:
            folderName = '%s/hists/%s/iter%d'%(base, self.name, self.curIteration)
        if os.path.isdir(folderName):
            if os.path.isdir(folderName+"_old"):
                shutil.rmtree("%s_old"%(folderName))
            os.rename(folderName, folderName+"_old")
        os.makedirs(folderName)
        
        if self.curIteration > 0 and not self.alignment.isDesign: # don't have to check for isDesign here because it always ends after iteration 0...
            shutil.copyfile('%s/hists/%s/iter%d/allData_iterationApe.root'%(base, self.name, self.curIteration-1),folderName+"/allData_iterationApe.root")
        fileNames = ['%s/hists/%s/%s%s.root'%(base, self.name, self.dataset.sampleType, str(i)) for i in range(1, self.dataset.nFiles+1)]
        fileString = " ".join(fileNames)
        
        from autoSubmitterTemplates import mergeTemplate
        merge_result = subprocess.call(mergeTemplate.format(path=folderName, inputFiles=fileString), shell=True) # returns exit code (0 if no error occured)
        for name in fileNames:
            os.remove(name)
            
        if os.path.isfile("%s/allData.root"%(folderName)) and merge_result == 0:
            self.status = STATE_MERGE_DONE
        else:
            self.status = STATE_MERGE_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        self.print_status()
    
    def do_summary(self):
        self.status = STATE_SUMMARY_WAITING        
        from autoSubmitterTemplates import summaryTemplate
        if self.alignment.isDesign:
            inputCommands = "iterNumber=%d setBaseline=%r measurementName=%s baselineName=%s"%(self.curIteration,self.alignment.isDesign,self.name, self.name)
        else:
            inputCommands = "iterNumber=%d setBaseline=%r measurementName=%s baselineName=%s"%(self.curIteration,self.alignment.isDesign,self.name, self.alignment.baselineDir)
        
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
        inputCommands = "iterNumber=%d setBaseline=%r measurementName=%s"%(self.curIteration,self.alignment.isDesign,self.name)

        local_setting_result = subprocess.call(localSettingTemplate.format(inputCommands=inputCommands), shell=True) # returns exit code (0 if no error occured)
        if local_setting_result == 0:
            self.status = STATE_LOCAL_DONE
        else:
            self.status = STATE_LOCAL_FAILED
            self.finishTime = subprocess.check_output(["date"]).strip()
        self.print_status()
    def finish_iteration(self):
        print("APE Measurement %s just finished iteration %d"%(self.name, self.curIteration))
        if self.curIteration < self.maxIterations:
            self.curIteration += 1
            self.status = STATE_ITERATION_START
        else:
            self.status = STATE_FINISHED
            self.finishTime = subprocess.check_output(["date"]).strip()
            print("APE Measurement %s, which was started at %s was finished after %d iterations, at %s"%(self.name, self.startTime, self.curIteration, self.finishTime))
    def kill(self):
        from autoSubmitterTemplates import killJobTemplate
        for job in self.runningJobs:
            subprocess.call(killJobTemplate.format(jobName=job), shell=True)
        self.runningJobs = []
        self.status = STATE_NONE
    def purge(self):
        self.kill()
        folderName = '%s/hists/%s'%(base, self.name)
        shutil.rmtree(folderName)

        
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
    parser.add_argument("-o", "--one", action="store_true", dest="one_iteration", default=False,
                          help="Do only one loop iteration and then quit")
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
                        print("Measurement %s in state %s in iteration %d was resumed"%(res.name, res.get_status(), res.curIteration))
                        # Killing and purging is done here, because it doesn't make 
                        # sense to kill or purge a measurement that was just started
                        for to_kill in args.kill:
                            if res.name == to_kill:
                                res.kill()
                        for to_purge in args.purge:
                            if res.name == to_purge:
                                res.purge()
            except IOError:
                print("Could not resume because %s could not be opened, exiting"%(pickle_name))
                sys.exit()
            
    # read out from config file
    if args.configs != []:
        config = ConfigParser.RawConfigParser()
        config.optionxform = str 
        config.read(args.configs)
    
        for name, opts in config.items("measurements"):
            if name in map(lambda x: x.name ,measurements):
                print("Error: APE Measurement with name %s already exists, skipping"%(name))
                continue
            
            settings = opts.split(" ")
            if len(settings) < 2:
                print("Error: number of arguments for APE Measurement %s is insufficient"%(name))
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
            
            print("APE Measurement %s was started"%(measurement.name))
            
    
    while True:
        measurements = [measurement for measurement in measurements if not measurement.status==STATE_NONE]
        save(measurements)
        
        if len(measurements) == 0:
            print("No APE measurements are active, exiting")
            break
        for measurement in measurements:
            if measurement.status == STATE_ITERATION_START:
                # start bjobs
                print("APE Measurement %s just started iteration %d"%(measurement.name, measurement.curIteration))
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
                print("APE Measurement %s just started iteration %d"%(measurement.name, measurement.curIteration))
                measurement.submit_jobs()
                save(measurements)
        if args.one_iteration:
            break
            
        
        
        time_remaining = clock_interval
        while time_remaining > 0:
            print("Sleeping for %s seconds, you can safely [CTRL+C] now"%(time_remaining))
            time.sleep(1)
            time_remaining -= 1
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        print("")
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
            
if __name__ == "__main__":
    main()
