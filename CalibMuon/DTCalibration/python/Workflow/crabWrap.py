import os,time,sys
from crab import Crab,common,parseOptions,CrabException
from crabStatusFromReport import crabStatusFromReport

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

"""
# Add method to Status classes
import Status
import StatusServer
Status.Status.summary = summaryStandAlone
StatusServer.StatusServer.summary = summaryServer
"""

def crabAction(options, action = None):

    options = parseOptions(options)

    crab = Crab()
    result = None
    try:
        crab.initialize_(options)
        crab.run()
        if action: result = action(crab)
        del crab
        print 'Log file is %s%s.log'%(common.work_space.logDir(),common.prog_name) 
    except CrabException, e:
        del crab
        #print '\n' + common.prog_name + ': ' + str(e) + '\n' 
        raise
        
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

def crabStatus(project):
    options = ['-status']
    if project:
        options.append('-c')
        options.append(project)

    def action(crab):
        #act = '-status'
        #return crab.actions[act].summary()
        xml = crab.cfg_params.get("USER.xml_report",'')
        return common.work_space.shareDir() + xml
        
    xmlreport = crabAction(options,action)
    status = crabStatusFromReport(xmlreport)
 
    return status

def convertStatus(status):
    """
    doneStatus = ['Done','Done (success)','Cleared','Retrieved']
    failedStatus = ['Aborted','Done (failed)','Killed','Cancelled']
    ignoreStatus = ['Created']
    """
    doneStatus = ['SD','E']
    failedStatus = ['A','DA','K']
    runningStatus = ['R']
    ignoreStatus = ['C']
    sumDone = 0.0
    sumFailed = 0.0
    sumRunning = 0.0
    sumIgnore = 0.0
    for key in status:
        if key in doneStatus: sumDone += status[key]
        if key in failedStatus: sumFailed += status[key]
        if key in runningStatus: sumRunning += status[key]
        if key in ignoreStatus: sumIgnore += status[key]

    # frac(done)' = N*frac(done)/(N - N*frac(ignore)) = frac(done)/(1 - frac(ignore))
    fracDone = 100.0*sumDone/(100.0 - sumIgnore)
    fracFailed = 100.0*sumFailed/(100.0 - sumIgnore)
    fracRun = 100.0*sumRunning/(100.0 - sumIgnore)

    result = {'Finished':fracDone,
              'Failed':fracFailed,
              'Running':fracRun}

    return result 

def checkStatus(project, threshold = 95.0):

    status = crabStatus(project)
 
    print "Percentage of jobs per status:"
    maxLength = max( [len(x) for x in status] )
    for item in status:
        print "%*s: %.0f%%" % (maxLength,item,status[item])


    statusNew = convertStatus(status)
       
    print "Relative percentage finished: %.0f%%" % statusNew['Finished']
    print "Relative percentage failed  : %.0f%%" % statusNew['Failed']
    print "Relative percentage running : %.0f%%" % statusNew['Running']

    finished = False
    # Condition for stopping
    #if fracFailed > 50.0: raise RuntimeError,'Too many jobs have failed (%.0f%%).' % fracFailed

    # Condition for considering it finished
    if statusNew['Finished'] >= threshold: finished = True 

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

def initCrabEnvironment():
    pythonpathenv = os.environ['PYTHONPATH']
    pythonpathbegin = pythonpathenv.split(':')[0].rstrip('/')
    pythonpathend = pythonpathenv.split(':')[-1].rstrip('/')

    indexBegin = sys.path.index(pythonpathbegin)
    if os.environ.has_key('CRABPSETPYTHON'): sys.path.insert( indexBegin, os.environ['CRABPSETPYTHON'] )
    if os.environ.has_key('CRABDLSAPIPYTHON'): sys.path.insert( indexBegin, os.environ['CRABDLSAPIPYTHON'] )
    if os.environ.has_key('CRABDBSAPIPYTHON'): sys.path.insert( indexBegin, os.environ['CRABDBSAPIPYTHON'] )

    if os.environ['SCRAM_ARCH'].find('32') != -1 and os.environ.has_key('CRABPYSQLITE'):
        sys.path.insert( indexBegin, os.environ['CRABPYSQLITE'] )
    elif os.environ['SCRAM_ARCH'].find('64') != -1 and os.environ.has_key('CRABPYSQLITE64'):
        sys.path.insert( indexBegin, os.environ['CRABPYSQLITE64'] )

    indexEnd = sys.path.index(pythonpathend) + 1
    if os.environ.has_key('CRABPYTHON'):
        if indexEnd >= len(sys.path): sys.path.append( os.environ['CRABPYTHON'] )
        else: sys.path.insert( indexEnd, os.environ['CRABPYTHON'] )

    #print sys.path

    #os.environ['LD_LIBRARY_PATH'] = os.environ['GLITE_LOCATION'] + '/lib' + ':' + os.environ['LD_LIBRARY_PATH']
    os.environ['VOMS_PROXY_INFO_DONT_VERIFY_AC'] = '1'
    #print os.environ['LD_LIBRARY_PATH']
    #print os.environ['VOMS_PROXY_INFO_DONT_VERIFY_AC'] 
    
    """ 
    export LD_LIBRARY_PATH=${GLITE_LOCATION}/lib:${LD_LIBRARY_PATH}
    export VOMS_PROXY_INFO_DONT_VERIFY_AC=1
    """
   
    ## Get rid of some useless warning
    try:
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        # import socket
        # socket.setdefaulttimeout(15) # Default timeout in seconds
    except ImportError:
        pass # too bad, you'll get the warning

    # Remove libraries which over-ride CRAB libs and DBS_CONFIG setting
    badPaths = []
    if os.environ.has_key('DBSCMD_HOME'): # CMSSW's DBS, remove last bit of path
        badPaths.append('/'.join(os.environ['DBSCMD_HOME'].split('/')[:-1]))
    if os.environ.has_key('DBS_CLIENT_CONFIG'):
        del os.environ['DBS_CLIENT_CONFIG']

    def pathIsGood(checkPath):
        """
        Filter function for badPaths
        """
        for badPath in badPaths:
            if checkPath.find(badPath) != -1:
                return False
        return True

    sys.path = filter(pathIsGood, sys.path)

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
