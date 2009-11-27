import time,sys
from crab import *
import common

def computeSummaryCRAB260(up_task):
    """
    Computes jobs summary for given task
    """
    taskId = str(up_task['name'])
    task_unique_name = str(up_task['name'])
    ended = None

    summary = {}
    nJobs = 0
    for job in up_task.jobs :
        id = str(job.runningJob['jobId'])
        jobStatus =  str(job.runningJob['statusScheduler'])
        jobState =  str(job.runningJob['state'])
        dest = str(job.runningJob['destination']).split(':')[0]
        exe_exit_code = str(job.runningJob['applicationReturnCode'])
        job_exit_code = str(job.runningJob['wrapperReturnCode'])
        ended = str(job['closed'])  
        printline=''
        if dest == 'None' :  dest = ''
        if exe_exit_code == 'None' :  exe_exit_code = ''
        if job_exit_code == 'None' :  job_exit_code = ''
        if job.runningJob['state'] == 'SubRequested' : jobStatus = 'Submitting'
        if job.runningJob['state'] == 'Terminated': jobStatus = 'Done'
        
        if summary.has_key(jobStatus): summary[jobStatus] += 1
        else: summary[jobStatus] = 1 
        nJobs += 1

    for item in summary: summary[item] = 100.*summary[item]/nJobs 

    return summary

def computeSummaryCRAB251(up_task):
    "Computes jobs summary for given task" 
 
    taskId = str(up_task['name'])
    task_unique_name = str(up_task['name'])
    ended = None

    summary = {}
    nJobs = 0
    for job in up_task.jobs :
        id = str(job.runningJob['jobId'])
        jobStatus =  str(job.runningJob['statusScheduler'])
        dest = str(job.runningJob['destination']).split(':')[0]
        exe_exit_code = str(job.runningJob['applicationReturnCode'])
        job_exit_code = str(job.runningJob['wrapperReturnCode'])
        ended = str(job['standardInput'])  
        printline=''
        if dest == 'None' :  dest = ''
        if exe_exit_code == 'None' :  exe_exit_code = ''
        if job_exit_code == 'None' :  job_exit_code = ''
        #printline+="%-6s %-18s %-36s %-13s %-16s %-4s" % (id,jobStatus,dest,exe_exit_code,job_exit_code,ended)
        #print printline
        if summary.has_key(jobStatus): summary[jobStatus] += 1
        else: summary[jobStatus] = 1 
        nJobs += 1

    for item in summary: summary[item] = 100.*summary[item]/nJobs 

    return summary

computeSummary = computeSummaryCRAB260

def summaryStandAlone(self):
    """
    Returns jobs summary
    """
    task = common._db.getTask()
    upTask = common.scheduler.queryEverything(task['id'])
    return computeSummary(upTask)

def summaryServer(self):
    """
    Returns jobs summary
    """
    #self.resynchClientSide()
        
    upTask = common._db.getTask()  
    return computeSummary(upTask)

# Add method to Status classes
import Status
import StatusServer
Status.Status.summary = summaryStandAlone
StatusServer.StatusServer.summary = summaryServer

def crabActionCRAB260(options, action = None):

    options = parseOptions(options)

    crab = Crab()
    result = None
    try:
        crab.initialize_(options)
        crab.run()
        if action: result = action(crab)
        del crab
        #print 'Log file is %s%s.log'%(common.work_space.logDir(),common.prog_name)  
        #print '\n##############################  E N D  ####################################\n'
    except CrabException, e:
        del crab
        #print '\n' + common.prog_name + ': ' + str(e) + '\n'
        pass
    pass
    if (common.logger): common.logger.delete()

    if result: return result

def crabActionCRAB251(options, action = None):

    options = parseOptions(options)

    result = None
    try:
        crab = Crab(options)
        crab.run()
        common.apmon.free()
        if action: result = action(crab)
        del crab
        #print 'Log file is %s%s.log'%(common.work_space.logDir(),common.prog_name)  
        #print '\n##############################  E N D  ####################################\n'
    except CrabException, e:
        print '\n' + common.prog_name + ': ' + str(e) + '\n'
        pass
    pass
    #if (common.logger): common.logger.delete()

    if result: return result

crabAction = crabActionCRAB260

def crabCreate(dir = '.', crabCfg_name = 'crab.cfg'):

    cwd = os.getcwd()
    os.chdir(dir)

    options = ['-create','-cfg',crabCfg_name]

    project = crabAction(options,lambda crab: common.work_space.topDir())

    os.chdir(cwd)

    return project

def crabSubmit(project):
    options = ['-submit','-c',project]

    crabAction(options)

    return

def checkStatus(project, threshold = 95.0):
    options = ['-status']
    if project:
        options.append('-c')
        options.append(project)

    def action(crab):
        act = '-status'
        return crab.actions[act].summary()

    status = crabAction(options,action)
    print "Percentage of jobs per status:"
    for item in status:
        print "%s %.2f"%(item,status[item])

    finished = False
    #if status.has_key('Done') and status['Done'] > threshold: finished = True
    if status.has_key('Done') and status['Done'] > 50.0:
        ignoreStatus = ['Created']
        sum = 0.0
        for item in ignoreStatus:
            if status.has_key(item): sum += status[item]

        # frac(done)' = N*frac(done)/(N - N*frac(ignore)) = frac(done)/(1 - frac(ignore))
        per_new = 100.0*status['Done']/(100.0 - sum)
        if per_new > threshold: finished = True 

    return finished

def getOutput(project):
    options = ['-getoutput']
    if project:
        options.append('-c')
        options.append(project)

    crabAction(options)

    return

def crabWatch(action,project = None, threshold = 95.0):
    #for i in range(5):
    while True:
        if checkStatus(project,threshold): break
        time.sleep(180)
 
    print "Finished..."

    action(project)
  
    return

def run(project = None, threshold = 95.0):

    crabWatch(getOutput,project,threshold)
    
    return

if __name__ == '__main__':
    project = None
    threshold = 95.0
    for opt in sys.argv:
        if opt[:8] == 'project=':
            project = opt[8:]
            print "Running on CRAB project",project
        if opt[:10] == 'threshold=':
            threshold = float(opt[10:])
            print "Using threshold",threshold 
    
    run(project,threshold)
