#! /usr/bin/env python

import os
import time
import sys

commands_file=open('cmsDriver_commands.txt','r')
lines=commands_file.readlines()
commands_file.close()

commands=[]

for line in lines:
    if line[0]=='$':
        commands.append(line[:-1])
        
nfail=0
npass=0
report=''

for command in commands:
    print 'Preparing to run %s' %command 
    
    commandbase=''
    for word in command.split(' ')[1:]:
        commandbase+='%s_'%word
    logfile='%s.log' %commandbase[:-1]
    
    startime='date %s' %time.asctime()
    executable='%s > %s 2>&1' %(command,logfile)
    #print executable
    exitcode=os.system(executable)
    #exitcode=0
    endtime='date %s' %time.asctime()
    tottime='%s-%s'%(endtime,startime)
    
    if exitcode!=0:
        log='%s : FAILED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
        print log
        report+='%s\n'%log
        nfail+=1
    else:
        log='%s : PASSED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
        print log
        report+='%s\n'%log
        npass+=1    
    
report+='\n %s tests passed, %s failed \n' %(npass,nfail)

print report

runall_report_name='runall-report.log'
runall_report=open(runall_report_name,'w')
runall_report.write(report)
runall_report.close()