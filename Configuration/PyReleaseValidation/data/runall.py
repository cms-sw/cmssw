#! /usr/bin/env python3

from __future__ import print_function
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
        logfile = logfile.replace('/','_') # otherwise the path in the args to --cusotmize make trouble
        
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
        opts, args = getopt.getopt(argv, "", ["nproc=","dohighstat",'hlt','inFile=','intbld'])
    except getopt.GetoptError as e:
        print("unknown option", str(e))
        sys.exit(2)
        
# check command line parameter
    np=1
    doHighStat=0
    hlt = False
    inFile = None
    intBld = False
    for opt, arg in opts :
        if opt == "--inFile" :
            inFile=arg
        if opt == "--nproc" :
            np=arg
        if opt == "--dohighstat" :
            doHighStat=1
        if opt in ('--hlt',): # note: trailing comma needed for single arg to indicate tuple
            hlt = True
        if opt in ('--intbld',): # note: trailing comma needed for single arg to indicate tuple
            intBld = True

    if hlt:
        print("\nWARNING: option --hlt is deprecated as this is now default.\n")

    if inFile:
        commands_standard_file=open(inFile,'r')
        lines_standard=commands_standard_file.readlines()
        commands_standard_file.close()
        lines=lines_standard
    else:
        commands_standard_file=open('cmsDriver_standard_hlt.txt','r')
        lines_standard=commands_standard_file.readlines()
        commands_standard_file.close()
        lines=lines_standard

        if doHighStat==1:
            commands_highstat_file=open('cmsDriver_highstats_hlt.txt','r')
            lines_highstat=commands_highstat_file.readlines()
            commands_highstat_file.close()

            lines=lines+lines_highstat
   

    # for the integration builds, check only these samples:
    forIB = [ # from the standard_hlt:
             'SingleMuPt10', 'SinglePiPt1', 'SingleElectronPt10', 'SingleGammaPt10',
             'MinBias', 'QCD_Pt_80_120', 'ZEE', 'BJets_Pt_50_120','TTbar',
             # from the highstats_hlt
             'SinglePiE50HCAL', 'H130GGgluonfusion', 'QQH120Inv', 'bJpsiX', 
             'JpsiMM', 'BsMM', 'UpsMM', 'CJets_Pt_50_120'
             ]
    
    commands=[]
    for line in lines:
        if ( line[0]!='#' and
           line.replace(' ','')!='\n' ):
               linecomponents=line.split('@@@')
               if intBld and linecomponents[0].strip() not in forIB: continue
               command=linecomponents[1][:-1]
               commands.append(command)
               print('Will do: '+command)
        

    nfail=0
    npass=0
    report=''

    clist = []
    cdone = []
    i=0
    print('Running in %s thread(s)' %np)

    for command in commands:
        print('Preparing to run %s' %command) 
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
                isA=pingle.is_alive()
                if ( isA ): i+=1
                if ( not isA and mystat==0 ): 
                    nfail+=pingle.nfail
                    npass+=pingle.npass
                    report+=pingle.report
                    cdone[j]=1
                    print(pingle.report)
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
            print(pingle.report)
        
    report+='\n %s tests passed, %s failed \n' %(npass,nfail)
    print(report)
    
    runall_report_name='runall-report.log'
    runall_report=open(runall_report_name,'w')
    runall_report.write(report)
    runall_report.close()
    
    if hlt:
        print("\nWARNING: option --hlt is deprecated as this is now default.\n")

if __name__ == '__main__' :
    main(sys.argv[1:])
