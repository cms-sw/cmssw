#!/usr/bin/env python -u

import os, sys, re, time, commands, glob
from optparse import OptionParser, OptionGroup
from threading import Thread
import random
import smtplib
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders

def sort_dict(adict):
    keys = sorted(adict.keys())
    return [adict[key] for key in keys]

import xml.dom.minidom
import re

class FWJRParser:
    Run='RunProduct'
    LS='LumiProduct'
    def __init__(self, file):
        self._file = file
        self._fwjr = xml.dom.minidom.parse(self._file)
        self._blacklist = ['FileName', 'FileClass', 'DumpType', 'Source', 'pathNameMatch', 'Type']
    def getText(self, nodelist):
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)

    def handleReport(self, report):
        result = {}
        subsys = 'Fake'
        for node in report.childNodes:
            if node.nodeName in self._blacklist or not node.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                continue
            m = re.match('(.*?)_(.*)', node.nodeName)
            if m:
                if len(result.keys()) >= 1 and m.group(1) not in result.keys():
                    result[subsys] += '}'
                subsys = m.group(1)
                if subsys not in result.keys():
                    result[subsys] = "{'subsys': '%s', '%s': '%s'" % (m.group(1),m.group(2), node.getAttribute('Value'))
                else:
                    result[subsys] += ", '%s': '%s'" % (m.group(2), node.getAttribute('Value'))
        if subsys in result.keys():
            result[subsys] += '}'
        return result

    def getReport(self, type):
        """Return an object that contains complete information about
        the selected type of products, either Run or Lumi. The key is
        the directory name, the value is an object itself with several
        keys, one for each kind of monitored quantity (MB, b, b_h, h, Kb_h)."""
        
        dqmReports = self._fwjr.getElementsByTagName('AnalysisFile')
        for report in dqmReports:
            reportType = report.getElementsByTagName("Type")
            if len(reportType) > 0:
                if reportType[0].getAttribute('Value') == type:
                    return self.handleReport(report)


class ThreadManager:
    def __init__(self):
        self.threadList = []

    def activeThreads(self):
        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1
        return nActive

    def addThread(self, a_thread):
        self.threadList.append(a_thread)

class TestRunner(Thread):
    def __init__(self, mainDir, timestamp, testNumber, cwd, forceTest, fakeRun, Num=0, Source='', Client=''):
        Thread.__init__(self)

        self.mainDir = mainDir
        self.timestamp = str(timestamp)
        self.testNumber = str(testNumber)
        self.cwd = cwd
        self.forceTest = forceTest
        self.fakeRun = fakeRun
        self.outputState = 0
        self.sequence = ['a', 'b', 'c']
        self.source = Source
        self.client = Client
        self.numev = Num
        return

    def inspect(self):
        return "inspect: " + self.timestamp + " " +  self.testNumber + " " + os.getcwd() + '\n'

    def run(self):
        typeCheck = ['b', 'MB', 'h']
        wdir = self.cwd + '/' + self.timestamp + '/' + self.testNumber
       
        m = re.search('(CMSSW.*?)/src.*', self.cwd)
        if m:
            cmssw_version = m.group(1).replace('-', '_')
        else:
            cmssw_version = 'Unknown'
        print 'Performing test %s on directory %s' % (self.testNumber, wdir)
        # do not perform actual test if already done at the same time
        # and with the same test number.
        if os.path.exists(wdir):
            if self.forceTest == 0:
                return
        else:
            os.makedirs(wdir)
        self.inspect()
        if self.fakeRun == 0:
            logfile = wdir + '/thread.log'
            l = open(logfile, 'w', 0)
            # copy driver{testNumber}{sequence}.sh command files
            for s in range(0,len(self.sequence)):
                test = 'driver' + self.testNumber + self.sequence[s] + '.sh'
                l.write(time.ctime(time.time()) + ' cp -p %s %s ' % (self.mainDir+'/'+test, wdir+'/'+test))
                l.write('\n')
                (status,output) = commands.getstatusoutput('cp -p %s %s ' % (self.mainDir+'/'+test, wdir+'/'+test))
                if self.numev != 0:
                    print 'Customizing Event to process/generate %d' % self.numev
                    print "sed -i -e '#^numev.*#numev=%d#' %s/%s " % (self.numev, wdir, test)
                    (status,output) = commands.getstatusoutput("sed -i -e 's#^numev.*#numev=%d#' %s/%s " % (self.numev, wdir, test))
                if self.source and s==0:
                    print 'Customizing source sequence with %s' % self.source
                    print "sed -i -e '#^DQMSEQUENCE.*#DQMSEQUENCE=%s#' %s/%s " % (self.source, wdir, test)
                    (status,output) = commands.getstatusoutput("sed -i -e 's#^DQMSEQUENCE.*#DQMSEQUENCE=%s#' %s/%s " % (self.source, wdir, test))
                if self.client and (s==1 or s==2):
                    print 'Customizing client sequence with %s' % self.client
                    (status,output) = commands.getstatusoutput("sed -i -e 's#^DQMSEQUENCE.*#DQMSEQUENCE=%s#\' %s/%s " % (self.client, wdir, test))
                l.write(time.ctime(time.time()) + ' cd %s && source %s/%s' % (wdir, wdir, test))
                (status,output) = commands.getstatusoutput('cd %s && source %s/%s' % (wdir, wdir, test) )
                l.write(time.ctime(time.time()) + "After running CMSSW, this is the exit code " + str(status))
                l.write(time.ctime(time.time()) + '\n ' + output)
                l.write('\n')
                l.flush()
                if not status == 0:
                    l.write(self.inspect())
                    l.write('%s: Exit status different from 0, quitting. Status: %d' % (time.ctime(time.time()), status) )
                    l.write('\n')
                    l.close()
                    f = open(self.cwd+'/'+self.testNumber+'_Failed.log', 'w')
                    f.write('\n')
                    f.close()
                    return
            l.close()
        f = open(self.cwd+'/'+self.testNumber+'_OK.log', 'w')
        xmls = glob.glob('%s/*.xml' % wdir)
        #xmls = commands.getoutput('ls -1tr %s/*.xml' % wdir ).split('\n')
        output = commands.getoutput('eval `scramv1 runtime -sh` && showtags -r | grep "^V"').split('\n')
        testPackages = ""
        for line in output:
            fields = re.split('\s*', line)
            if len(fields) >  1:
                testPackages += "{'package': '%s', 'current':'%s', 'base':'%s'}," %(fields[2], fields[0], fields[1])
        for single in xmls:
            filename = "%s/%s" %(self.cwd, single.split('/')[-1])
            print 'Analyzing %s' % single
            fwjr = FWJRParser(single)
            resRun = fwjr.getReport(FWJRParser.Run)
            if resRun:
                for k in resRun.keys():
                    f.write("db.results.save({'ts': '%s',  'IB': '%s', 'test': '%s', 'type': 'runProduct', 'logfile':'%s', 'packages': [%s], 'results': %s})\n " % (self.timestamp,cmssw_version,self.testNumber, filename.replace('FrameworkJobReport_',''),testPackages, resRun[k]))
            resLS = fwjr.getReport(FWJRParser.LS)
            if resLS:
                for k in resLS.keys():
                    f.write("db.results.save({'ts': '%s',  'IB': '%s', 'test': '%s', 'type': 'lsProduct', 'logfile':'%s', 'packages': [%s], 'results': %s})\n " % (self.timestamp,cmssw_version,self.testNumber, filename.replace('FrameworkJobReport_',''),testPackages, resLS[k]))
            # Now do memory checks aganist reference, if it exists
            if os.path.isfile(filename):
                fwjr = FWJRParser(filename)
                refRun = fwjr.getReport(FWJRParser.Run)
                if refRun:
                    for k in refRun:
                        if k in resRun:
                            res = eval(resRun[k])
                            ref = eval(refRun[k])
                            for kind in typeCheck:
                                self.checkMemAgainstRef(k, kind, float(ref[kind]), float(res[kind]), filename)
                    # Now look for possible new folders that are missing from the reference file.
                    for k in resRun:
                        if not k in refRun:
                            res = eval(resRun[k])
                            for kind in typeCheck:
                                self.checkMemAgainstRef(k, kind, 0., float(res[kind]), filename)
                            
        f.write('\n')
        f.close()
        return

    def checkMemAgainstRef(self, subsys, type, ref, res, filename):
        if res != ref:
            apath = '/'+'/'.join(filename.split('/')[1:-1])
            repFile = '%s/%s_MemoryChanged.log' % (apath,filename.split('/')[-1].split('_')[1])
            repLog = open(repFile,'a')
            if res > ref:
                repLog.write('%s FAILS on %s: Reference %f, current %f, diff %f\n' % (subsys, type, ref, res, (res-ref)))
            else:
                repLog.write('%s GAINS on %s: Reference %f, current %f, diff %f\n' % (subsys, type, ref, res, (ref-res)))
            repLog.close()
        

class TestOptionParser:
    def __init__(self):
        self.parser = OptionParser()
        self.group_general = OptionGroup(self.parser, "General Options",
                                         ""
                                         )
    def parseOptions(self):
        (self.options, self.args) = self.parser.parse_args()

    def error(self, message):
        self.parser.error(message)

    def f(self):
        return self.options.f

    def j(self):
        return self.options.j

    def n(self):
        return self.options.n

    def k(self):
        return self.options.k

    def s(self):
        return self.options.s

    def q(self):
        return self.options.q

    def noMail(self):
        return self.options.noMail



    def defineOptions(self):
        # GENARAL OPTIONS

        self.group_general.add_option("-j",
                                      dest="j",
                                      help="Number of parallel threads to run(one thread means one job)",
                                      action="store",
                                      type="int",
                                      metavar="THREADS")
        self.group_general.add_option("-n",
                                      dest="n",
                                      help="Specific list of tests to be submitted",
                                      action="store",
                                      type="string",
                                      metavar="TESTNUMBER")
        self.group_general.add_option("-f",
                                      dest="f",
                                      help="Force a specific directory into which the tests will be run.",
                                      action="store",
                                      type="int",
                                      metavar="TIMESTAMP")
        self.group_general.add_option("-k",
                                      dest="k",
                                      help="Simulate a run, simply recycle results from previous test. Used for internal debugging.",
                                      action="store_true")
        self.group_general.add_option("-s",
                                      dest="s",
                                      help="Specify a custom source and client sequence to run, comma separated",
                                      type="string",
                                      action="store")
        self.group_general.add_option("-q",
                                      dest="q",
                                      help="Specify a custom number of events to process",
                                      type="int",
                                      action="store")
        self.group_general.add_option("--noMail",
                                      dest="noMail",
                                      help="Stop the email sender",
                                      action="store_true")
        self.parser.add_option_group(self.group_general)


def notifyMe(mailFile):

    """Send an email notification to the appropriate recipient with
    fixed SUBJECT(Latest IB) and the specified message body."""

    me = 'marco.rovere@cern.ch'
    me_bis = 'federico.de.guio@cern.ch'
    f = open(mailFile)
    msg = email.message_from_file(f)
    f.close()
    msg['Subject'] = 'Latest IB'
    msg['From']    = me
    msg['To']      = me

    s=smtplib.SMTP("localhost")
    s.sendmail(me, [me], msg.as_string())
    s.quit()
    s=smtplib.SMTP("localhost")
    s.sendmail(me_bis, [me_bis], msg.as_string())
    s.quit()

def send_mail(attachement):

    server = "localhost"
    me = 'marco.rovere@cern.ch'
    me_bis = 'federico.de.guio@cern.ch'
    msg = MIMEMultipart()
    msg['From'] = me
    msg['To']   = me
    msg['Subject'] = 'Latest IB done by %s' % os.environ['USER'] 

    msg.attach( MIMEText('Tests Done.') )

    part = MIMEBase('application', "octet-stream")
    part.set_payload( open(attachement,"rb").read() )
    Encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(attachement))
    msg.attach(part)

    smtp = smtplib.SMTP(server)
    smtp.sendmail(me, [me], msg.as_string())
    smtp.close()
    smtp = smtplib.SMTP(server)
    smtp.sendmail(me_bis, [me_bis], msg.as_string())
    smtp.close()



if __name__ == '__main__':
    try:
        numTests = 10
        tm = ThreadManager()
        ts = int(time.time())
        mainDir = os.path.dirname(os.path.realpath(__file__))
        cwd = os.path.dirname(os.path.realpath(__file__))#os.getcwd()
        parser = TestOptionParser()
        parser.defineOptions()
        parser.parseOptions()
        threads = 1
        forceTest = 0
        fakeRun = 0
        source = ''
        client = ''
        numev = 0


        if parser.q():
            numev = parser.q()
        if parser.s():
            m = re.match('(.*),(.*)', parser.s())
            if m:
                source = m.group(1)
                client = m.group(2)
            else:
                print "Error, wrong DQM sequence format supplied: source_sequence,client_sequence."
                sys.exit(1)
        if parser.f():
            ts = parser.f()
            forceTest = 1
        if parser.k():
            fakeRun = 1
        if parser.j():
            threads = parser.j()
            print "Running ", threads, " threads in parallel."
        if parser.n():
            jobs = parser.n().split(',')
            for test in jobs:
                # make sure we don't run more than the allowed number of threads:
                while True:
                    print 'Active threads: ', tm.activeThreads()
                    if tm.activeThreads() < threads:
                        break
                    print 'Sleeping...'
                    time.sleep(10)

                a_thread = TestRunner(mainDir,ts,test, cwd, forceTest, fakeRun, Num=numev, Source=source, Client=client)
                a_thread.inspect()
                tm.addThread(a_thread)
                a_thread.start()
                time.sleep(random.randint(1,5)) # try to avoid race cond by sleeping random amount of time [1,5] sec

        while tm.activeThreads() > 0 :
            time.sleep(10)

        mailFile = '%s/%s' %  (cwd,'MailMessage.txt')
        f = open(mailFile, 'w')

        # Now append tags information
        if os.path.exists('%s' % cwd ):
            tags = glob.glob('*tags')
            for t in tags:
                f.write('Analizing latest IB with TAGS:\n')
                ff = open(t)
                for line in ff:
                    f.write(line)
                ff.close()

        # Now parse all log files, showtags info and send around 1 gigantic email...
        print "Assembling final report on %s" % cwd
        if os.path.exists('%s' % cwd ):
            putAtTheEnd = ''
            command = 'find %s/*log -cnewer %s/%d | sort -n -k 1 -t _' % (cwd, cwd, ts)
            print 'Running command %s' % command
            logs = commands.getoutput(command).split('\n')
            for l in logs:
                print "Processing log file: %s" % l
                f.write('****************** %s' % l )
                ff = open(l)
                archiveIt = 0
                for line in ff:
                    if re.search('\+\+\+ Cut to here \+\+\+.*', line):
                        archiveIt = 0
                        continue
                    if archiveIt == 1:
                        if not re.search('\+\+\+ Cut from here \+\+\+.*', line):
                            putAtTheEnd += line
                    if re.search('\+\+\+ Cut from here \+\+\+.*', line):
                        archiveIt = 1
                        continue
                    if not archiveIt == 1:
                        if not re.search('\s+->\s+.*', line):
                            f.write(line)
                ff.close()
            f.write(putAtTheEnd)
        f.close()

        #moving the logs in the unix time folder
        for fname in glob.glob(cwd+"/*.log"):
            list = fname.split("/")
            os.rename(fname,cwd+"/"+str(ts)+"/"+list[-1])

        if not parser.noMail():
            send_mail(mailFile)
    except:
        print "Error executing whiteRabbit.py: exiting."
        sys.exit(1)
