
from threading import Thread

from Configuration.PyReleaseValidation import WorkFlow
import os,time
import shutil
from subprocess import Popen 
from os.path import exists, basename, join
from os import getenv
from datetime import datetime
from hashlib import sha1
import urllib2, base64, json, re
from socket import gethostname

# This is used to report results of the runTheMatrix to the elasticsearch
# instance used for IBs. This way we can track progress even if the logs are
# not available.
def esReportWorkflow(**kwds):
  # Silently exit if we cannot contact elasticsearch
  es_hostname = getenv("ES_HOSTNAME")
  es_auth = getenv("ES_AUTH")
  if not es_hostname and not es_auth:
    return
  payload = kwds
  sha1_id = sha1(kwds["release"] + kwds["architecture"] +  kwds["workflow"] + str(kwds["step"])).hexdigest()
  d = datetime.now()
  if "_201" in kwds["release"]:
    datepart = "201" + kwds["release"].split("_201")[1]
    d = datetime.strptime(datepart, "%Y-%m-%d-%H00")
    payload["release_queue"] = kwds["release"].split("_201")[0]
  payload["release_date"] = d.strftime("%Y-%m-%d-%H00")
  # Parse log file to look for exceptions, errors and warnings.
  logFile = payload.pop("log_file", "")
  exception = ""
  error = ""
  errors = []
  inException = False
  inError = False
  if exists(logFile):
    lines = file(logFile).read()
    payload["message"] = lines
    for l in lines.split("\n"):
      if l.startswith("----- Begin Fatal Exception"):
        inException = True
        continue
      if l.startswith("----- End Fatal Exception"):
        inException = False
        continue
      if l.startswith("%MSG-e"):
        inError = True
        error = l
        error_kind = re.split(" [0-9a-zA-Z-]* [0-9:]{8} CET", error)[0].replace("%MSG-e ", "")
        continue
      if inError == True and l.startswith("%MSG"):
        inError = False
        errors.append({"error": error, "kind": error_kind})
        error = ""
        error_kind = ""
        continue
      if inException:
        exception += l + "\n"
      if inError:
        error += l + "\n"

  if exception:
    payload["exception"] = exception
  if errors:
    payload["errors"] = errors
      
  payload["hostname"] = gethostname()
  url = "https://%s/ib-matrix.%s/runTheMatrix-data/%s" % (es_hostname,
                                                          d.strftime("%Y-%W-1"),
                                                          sha1_id)
  request = urllib2.Request(url)
  if es_auth:
    base64string = base64.encodestring(es_auth).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
  request.get_method = lambda: 'PUT'
  data = json.dumps(payload)
  try:
    result = urllib2.urlopen(request, data=data)
  except urllib2.HTTPError, e:
    print e
    try:
      print result.read()
    except:
      pass

class WorkFlowRunner(Thread):
    def __init__(self, wf, noRun=False,dryRun=False,cafVeto=True,dasOptions="",jobReport=False, nThreads=1):
        Thread.__init__(self)
        self.wf = wf

        self.status=-1
        self.report=''
        self.nfail=0
        self.npass=0
        self.noRun=noRun
        self.dryRun=dryRun
        self.cafVeto=cafVeto
        self.dasOptions=dasOptions
        self.jobReport=jobReport
        self.nThreads=nThreads
        
        self.wfDir=str(self.wf.numId)+'_'+self.wf.nameId
        return

    def doCmd(self, cmd):

        msg = "\n# in: " +os.getcwd()
        if self.dryRun: msg += " dryRun for '"
        else:      msg += " going to execute "
        msg += cmd.replace(';','\n')
        print msg

        cmdLog = open(self.wfDir+'/cmdLog','a')
        cmdLog.write(msg+'\n')
        cmdLog.close()
        
        ret = 0
        if not self.dryRun:
            p = Popen(cmd, shell=True)
            ret = os.waitpid(p.pid, 0)[1]
            if ret != 0:
                print "ERROR executing ",cmd,'ret=', ret

        return ret
    
    def run(self):

        startDir = os.getcwd()

        if not os.path.exists(self.wfDir):
            os.makedirs(self.wfDir)
        elif not self.dryRun: # clean up to allow re-running in the same overall devel area, then recreate the dir to make sure it exists
            print "cleaning up ", self.wfDir, ' in ', os.getcwd()
            shutil.rmtree(self.wfDir) 
            os.makedirs(self.wfDir)

        preamble = 'cd '+self.wfDir+'; '
       
        realstarttime = datetime.now()
        startime='date %s' %time.asctime()

        # check where we are running:
        onCAF = False
        if 'cms/caf/cms' in os.environ['CMS_PATH']:
            onCAF = True

        ##needs to set
        #self.report
        self.npass  = []
        self.nfail = []
        self.stat = []
        self.retStep = []

        def closeCmd(i,ID):
            return ' > %s 2>&1; ' % ('step%d_'%(i,)+ID+'.log ',)

        inFile=None
        lumiRangeFile=None
        aborted=False
        for (istepmone,com) in enumerate(self.wf.cmds):
            # isInputOk is used to keep track of the das result. In case this
            # is False we use a different error message to indicate the failed
            # das query.
            isInputOk=True
            istep=istepmone+1
            cmd = preamble
            if aborted:
                self.npass.append(0)
                self.nfail.append(0)
                self.retStep.append(0)
                self.stat.append('NOTRUN')
                continue
            if not isinstance(com,str):
                if self.cafVeto and (com.location == 'CAF' and not onCAF):
                    print "You need to be no CAF to run",self.wf.numId
                    self.npass.append(0)
                    self.nfail.append(0)
                    self.retStep.append(0)
                    self.stat.append('NOTRUN')
                    aborted=True
                    continue
                #create lumiRange file first so if das fails we get its error code
                cmd2 = com.lumiRanges()
                if cmd2:
                    cmd2 =cmd+cmd2+closeCmd(istep,'lumiRanges')
                    lumiRangeFile='step%d_lumiRanges.log'%(istep,)
                    retStep = self.doCmd(cmd2)
                cmd+=com.das(self.dasOptions)
                cmd+=closeCmd(istep,'dasquery')
                retStep = self.doCmd(cmd)
                #don't use the file list executed, but use the das command of cmsDriver for next step
                # If the das output is not there or it's empty, consider it an
                # issue of this step, not of the next one.
                dasOutputPath = join(self.wfDir, 'step%d_dasquery.log'%(istep,))
                if not exists(dasOutputPath):
                  retStep = 1
                  dasOutput = None
                else:
                  # We consider only the files which have at least one logical filename
                  # in it. This is because sometimes das fails and still prints out junk.
                  dasOutput = [l for l in open(dasOutputPath).read().split("\n") if l.startswith("/")]
                if not dasOutput:
                  retStep = 1
                  isInputOk = False
                 
                inFile = 'filelist:' + basename(dasOutputPath)
                print "---"
            else:
                #chaining IO , which should be done in WF object already and not using stepX.root but <stepName>.root
                cmd += com
                if self.noRun:
                    cmd +=' --no_exec'
                if inFile: #in case previous step used DAS query (either filelist of das:)
                    cmd += ' --filein '+inFile
                    inFile=None
                if lumiRangeFile: #DAS query can also restrict lumi range
                    cmd += ' --lumiToProcess '+lumiRangeFile
                    lumiRangeFile=None
                # 134 is an existing workflow where harvesting has to operate on AlcaReco and NOT on DQM; hard-coded..    
                if 'HARVESTING' in cmd and not 134==self.wf.numId and not '--filein' in cmd:
                    cmd+=' --filein file:step%d_inDQM.root --fileout file:step%d.root '%(istep-1,istep)
                else:
                    if istep!=1 and not '--filein' in cmd:
                        cmd+=' --filein file:step%s.root '%(istep-1,)
                    if not '--fileout' in com:
                        cmd+=' --fileout file:step%s.root '%(istep,)
                if self.jobReport:
                  cmd += ' --suffix "-j JobReport%s.xml " ' % istep
                if self.nThreads > 1:
                  cmd += ' --nThreads %s' % self.nThreads
                cmd+=closeCmd(istep,self.wf.nameId)            
                
                esReportWorkflow(workflow=self.wf.nameId,
                                 release=getenv("CMSSW_VERSION"),
                                 architecture=getenv("SCRAM_ARCH"),
                                 step=istep,
                                 command=cmd,
                                 status="STARTED",
                                 start_time=realstarttime.isoformat(),
                                 workflow_id=self.wf.numId)
                retStep = self.doCmd(cmd)


            
            self.retStep.append(retStep)
            if retStep == 32000:
                # A timeout occurred
                self.npass.append(0)
                self.nfail.append(1)
                self.stat.append('TIMEOUT')
                aborted = True
            elif (retStep!=0):
                #error occured
                self.npass.append(0)
                self.nfail.append(1)
                if not isInputOk:
                  self.stat.append("DAS_ERROR")
                else:
                  self.stat.append('FAILED')
                #to skip processing
                aborted=True
            else:
                #things went fine
                self.npass.append(1)
                self.nfail.append(0)
                self.stat.append('PASSED')

            esReportWorkflow(workflow=self.wf.nameId,
                             release=getenv("CMSSW_VERSION"),
                             architecture=getenv("SCRAM_ARCH"), 
                             step=istep,
                             command=cmd,
                             status=self.stat[-1],
                             start_time=realstarttime.isoformat(),
                             end_time=datetime.now().isoformat(),
                             delta_time=(datetime.now() - realstarttime).seconds,
                             workflow_id=self.wf.numId,
                             log_file="%s/step%d_%s.log" % (self.wfDir, istep, self.wf.nameId))

        os.chdir(startDir)

        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)
        

        #### wrap up ####

        logStat=''
        for i,s in enumerate(self.stat):
            logStat+='Step%d-%s '%(i,s)
        self.report='%s_%s %s - time %s; exit: '%(self.wf.numId,self.wf.nameId,logStat,tottime)+' '.join(map(str,self.retStep))+'\n'

        return 



