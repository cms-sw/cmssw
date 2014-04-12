import os,sys,os.path,shlex
from subprocess import Popen,PIPE,STDOUT
from cStringIO import StringIO

def runCmmd(cmmdline,shell=False):
    """runsubprocess return processid
    """
    args=[]
    proc=Popen(cmmdline,shell=shell,stdout=PIPE,stdin=PIPE,stderr=STDOUT)
    stdout_value,stderr_value=proc.communicate()
    print repr(stdout_value)
    return proc.pid

def processRunning(processid):
    """
    check if a process is still running
    """
    return os.path.exists(os.path.join('/proc',str(processid)))

if __name__=='__main__':
    print processRunning(13378)
    pid= runCmmd('cat -; echo ";to stderr" 1>&2',shell=True)
    print processRunning(pid)
