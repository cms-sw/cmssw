#!/usr/bin/env python3
#this script:
#Find which Modules bad in the PCL and which are NOT bad in the DQM (express) (and dumps the info in a txt file)
#Find which modules are bad in the DQM (express) and which are still bad in DQM prompt (not because they are masked) (and dumps the info in a txt file) 
import copy
import re
import sys
from optparse import OptionParser 


def findpr(options):
    BadModpr=open(options.filenamePR,'r')
    bmpr=BadModpr.read()
    mod="Module"
    pcl="PCLBadModule"
    sub="SubDetector"


    prf =  re.findall(r'(SubDetector.*?\n\n.*?)(?:\n+^$|\Z)',bmpr,re.MULTILINE|re.DOTALL)
    prf =list(map(lambda x: re.split('\n+',x),prf))
    findpr.prd={}
    findpr.pralld={}
    # create dictionaries                                                                                                         
    prfd={}
   
    
    for k in prf:
        for l in k[1:]:
            n=re.split("\W+",l)
            prfd[n[1]]=(l)
        findpr.pralld[k[0]]=prfd
        prfd={}
            
    
    findpr.prd=copy.deepcopy(findpr.pralld)
    #dictionary with pclbadmodules only     

    for k in findpr.prd.keys():
            
        for l in findpr.prd[k].keys():
            if pcl not in findpr.prd[k][l]:
                findpr.prd[k].pop(l)
    
    #for k in findpr.pralld:
    #    print len(findpr.pralld[k])
    return 0

def findse(options):
    BadModse=open(options.filenameSE,'r')
    bmse=BadModse.read()

    sub="SubDetector"
    
    sef =  re.findall(r'(SubDetector.*?\n\n.*?)(?:\n+^$|\Z)',bmse,re.MULTILINE|re.DOTALL)
    sef =list(map(lambda x: re.split('\n+',x),sef))
    findse.sed={}
    
    sefd={}
    for k in sef:
        for l in k[1:]:
            n=re.split("\W+",l)
            sefd[n[1]]=(l)
        findse.sed[k[0]]=sefd
        sefd={}
      

    
    return 0
   

    
def printall():
    seFile=open('SEinPRBadMod.txt','w')
    prFile=open('PCLBadMod.txt','w')
    seFile.write("Bad Modules from stream express which are still bad in Prompt Reco\n\n")

    for x in findse.sed:
        seFile.write("\n"+x+"\n\n")
        for y in findse.sed[x]:
            if y in findpr.pralld[x]:
                seFile.write(findpr.pralld[x][y]+"\n")

    
    prFile.write("Bad Modules from Prompt Reco (PCLBadModules) that are not bad in Stream Express\n\n")
    
    for x in findpr.prd:
        prFile.write("\n"+x+"\n\n")
        for y in findpr.prd[x]:
             
            if y not in findse.sed[x]:
                
                prFile.write(findpr.prd[x][y]+"\n")
    
    return 0


############################################
if __name__ == "__main__":
    verbose = True
    usage = "useage: %prog [options] "
    parser = OptionParser(usage)
    parser.set_defaults(mode="advanced")
    parser.add_option("-p", "--filePR", type="string", dest="filenamePR", help="Get the name of the Prompt Reco file")
    parser.add_option("-s", "--fileSE", type="string", dest="filenameSE", help="Get the name of the Stream Express file")

    (options, args) = parser.parse_args()
    
    MyfilenamePR=findpr(options)
    MyfilenameSE=findse(options)
    Myprintall=printall()
        
        
    
