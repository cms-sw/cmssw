#! /usr/bin/env python

import os
import time
import sys
from threading import Thread

class testit(Thread):
    def __init__(self,command):
        Thread.__init__(self)
        self.command=command
        self.status=-1
        self.report=''
        self.nfail=0
        self.npass=0
    def run(self):
        commandbase=''
        for word in self.command.split(' ')[1:]:
            commandbase+='%s_'%word
        logfile='%s.log' %commandbase[:-1]
            
        startime='date %s' %time.asctime()
        executable='%s > %s 2>&1' %(self.command,logfile)
    
        exitcode=os.system(executable)
        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)
    
        if exitcode!=0:
            log='%s : FAILED - time: %s s - exit: %s\n' %(self.command,tottime,exitcode)
            self.report+='%s\n'%log
            self.nfail=1
            self.npass=0
        else:
            log='%s : PASSED - time: %s s - exit: %s\n' %(self.command,tottime,exitcode)
            self.report+='%s\n'%log
            self.nfail=0
            self.npass=1
                
def main(argv) :

    import getopt
    
    try:
        opts, args = getopt.getopt(argv, "", ["nproc="])
    except getopt.GetoptError:
        sys.exit(2)
        
# check command line parameter
    np=1
    for opt, arg in opts :
        if opt == "--nproc" :
            np=arg
        
    commands_file=open('cmsDriver_commands.txt','r')
    lines=commands_file.readlines()
    commands_file.close()
    
    commands=[]

    for line in lines:
        if line[0]!='#' and\
               line.replace(' ','')!='\n':
            linecomponents=line.split('@@@')
            command=linecomponents[1][:-1]
            commands.append(command)
            
        
    nfail=0
    npass=0
    report=''
    
    clist = []
    cdone = []
    i=0
    print 'Running in %s thread(s)' %np
    for command in commands:
        print 'Preparing to run %s' %command 
        current = testit(command)
        clist.append(current)
        cdone.append(0)
        current.start()

        i=int(np)
        while (int(i) >= int(np)): 
            i=0
            time.sleep(10)
            alen=len(cdone)
            for j in range(0,alen):
                mystat=cdone[j]
                pingle=clist[j]
                isA=pingle.isAlive()
                if ( isA ): i+=1
                if ( not isA and mystat==0 ): 
                    nfail+=pingle.nfail
                    npass+=pingle.npass
                    report+=pingle.report
                    cdone[j]=1
                    print pingle.report
#            print 'Number of running threads: %s' % i        

    alen=len(cdone)
    for j in range(0,alen):
        pingle=clist[j]
        mystat=cdone[j]
        if ( mystat == 0 ):  
            pingle.join()
            nfail+=pingle.nfail
            npass+=pingle.npass
            report+=pingle.report
            print pingle.report
        
    report+='\n %s tests passed, %s failed \n' %(npass,nfail)
    print report
    
    runall_report_name='runall-report.log'
    runall_report=open(runall_report_name,'w')
    runall_report.write(report)
    runall_report.close()
    

if __name__ == '__main__' :
    main(sys.argv[1:])
