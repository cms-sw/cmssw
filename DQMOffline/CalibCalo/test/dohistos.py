#! /usr/bin/env python3

from __future__ import print_function
import os
import sys
import errno

#print error/help message and exit
def help_message():
    print("Usage:\n\
dohistos [folder_name] [options] -v versions_to_compare -f files_to_compare\n\
Versions and files must be whitespace separated.\n\
If no folder is specified the pwd will be used.\n\
folder_name, if specified, must be the first argument.\n\
Options:\n\
--outfile out.root    specify the output root file name\n\
-h    prints this message\n\
--cvs    uses cvs access to retrieve edm2me_cfg.py\n\
-a    uses anonymous cvs access (only for debug purposes)\n\
--dbs    use dbs access to retrieve file paths, edit directly the source code to modify the query, while in --dbs mode don't use -f option\n\
--condition [IDEAL,STARTUP]    permits to specify the conditions when in dbs mode; if not specified IDEAL is default; must be coherent with --query option if specified\n\
-n [1000]   specify the number of events\n\
--like   additional like statement in the dbs query to narrow the number of dataset returned\n\
Example:\n\
./dohistos.py Folder01 -v CMSSW_X_Y_Z CMSSW_J_K_W -f file:/file1.root file:/file2.root")
    sys.exit()
#custom query is at the moment not useful:
#--query    with --dbs option specify the query passed to dbs cli (with quotes)\n\

#run command in the command line with specified environment
def runcmd(envir,program,*args):
    pid=os.fork()
    if not pid:
        os.execvpe(program,(program,)+args,envir)
    return os.wait()[0]

#print the help message
if "-h" in sys.argv or "-help" in sys.argv or "--help" in sys.argv:
    help_message()
   
#make the working directory (if specified) and move there
if len(sys.argv)>1:
    if not sys.argv[1][0]=="-":
        name=sys.argv[1]        
        try:
            os.mkdir(name)
        except OSError as inst:
            if inst.errno==errno.EEXIST:
                print("Warning: the specified working folder already exist")
        os.chdir(name)
else: help_message()

#read and parse the input
state="n"
like_query=""
num_evts="1000"
cvs=False
dbs=False
use_manual_num=False
anon=False#only for debug purposes
conditions='IDEAL_31X'
conditions_file='IDEAL'
#query_list=[]    #not useful
ver=[]
fil=[]
out_root="histo.root"
#used state letters (please keep updated): cflnqrv
for arg in sys.argv:
    if arg=="-v" or arg=="-V":
        state="v"
    elif arg=="-f" or arg=="-F":
        state="f"
    elif arg=="--outfile":
        state="r"
    elif arg=="--conditions":
        state="c"
    elif arg=="-n":
        state="n"
    #elif arg=="--query":    #not useful
    #    state="q"
    elif arg=="--cvs":
        cvs=True
    elif arg=="--dbs":
        dbs=True
    elif arg=="-a":#only for debug purposes
        anon=True
    elif arg=="--like":
        state="l"
############################################## state handling
    elif state=="v":
        ver.append(arg)
    elif state=="f":
        fil.append(arg)
    elif state=="r":
        out_root=arg
    elif state=="l":
        like_query=arg
    elif state=="c":
        conditions=arg
        usn=0
        for ncondt,condt in enumerate(arg):
            if condt=='_':
                usn=ncondt
                break
        conditions_file=conditions[:usn]
    elif state=="n":
        num_evts=arg
        use_manual_num=True
    #elif state=="q":    #not useful
    #    query_list.append(arg)

#check consistency of -f and --dbs
if len(fil)>0 and dbs:
    print("when using --dbs option, -f option is not needed")
    help_message()

###dbs query to retrieve the data with option --dbs
###|||||||||||||||||||||||||||||||||||||||||||||||||
dbsstr='python $DBSCMD_HOME/dbsCommandLine.py -c search --query="find dataset where phygrp=RelVal and primds=RelValMinBias and dataset.tier=GEN-SIM-DIGI-RAW-HLTDEBUG and file.name like *'+conditions+'* and file.name like *'+like_query+'* and file.release='
#dbsstr='dbs -c search --query="find dataset where phygrp=RelVal and primds=RelValMinBias and dataset.tier=GEN-SIM-DIGI-RAW-HLTDEBUG and file.name like *'+conditions+'* and file.release='
###|||||||||||||||||||||||||||||||||||||||||||||||||
dbsdataset=[]
dbsfile=[]
nevt=[]
nevent=0

#create folders and generate files
for nv,v in enumerate(ver):
    os.system("scramv1 project CMSSW "+v)
    os.chdir(v)
    env=os.popen("scramv1 runtime -sh","r")
    environment=os.environ
    for l in env.readlines():
        try:
            variable,value=l[7:len(l)-3].strip().split("=",1)
            environment[variable]=value[1:]
        except ValueError:
            print("Warning: environment variable problem")
    env.close()
    if cvs:
        if anon:#only for debug purposes, works only in cmsfarm
            os.system("eval `scramv1 runtime -sh`; source /cms-sw/slc4_ia32_gcc345/cms/cms-cvs-utils/1.0/bin/cmscvsroot.sh CMSSW; cvs login; addpkg DQMOffline/CalibCalo")
        else:
            runcmd(environment,"addpkg","DQMOffline/CalibCalo")

#dbs code
    if dbs:
        dbsdataset=[]
        dbsfile=[]
        nevt=[]
        #searching the required dataset
        inifil=False
        ris=os.popen(dbsstr+v+'"')
        for lnris in ris.readlines():
            print(lnris)
            if inifil:
                dbsdataset.append(lnris)
            else:
                #if lnris[:3]=="___":
                if lnris[:3]=="---":
                    inifil=True
        ris.close()
        dbsdataset=dbsdataset[2:]
        dbsdataset[0]=dbsdataset[0][0:-1]
        for lnris2 in dbsdataset:        
            print(lnris2)
        if len(dbsdataset)>1 or len(dbsdataset)==0:
            #print dbsdataset
            print("dbs search returned ",len(dbsdataset)," records, please modify the query so only one dataset is returned")
            sys.exit()
        else:
            #extracting the file names relative to the selected dataset
            inifil=False
            ris=os.popen('python $DBSCMD_HOME/dbsCommandLine.py -c search --query="find file where dataset like *'+dbsdataset[0]+'*"')
            for lnris in ris.readlines():
                if inifil:
                    dbsfile.append(lnris)
                else:
                    if lnris[:3]=="---":
                        inifil=True
            ris.close()
            dbsfile=dbsfile[2:]
            for dbsfn,dbsf in enumerate(dbsfile):
                dbsfile[dbsfn]=dbsfile[dbsfn][:-1]
            
            #extracting the total number of events    #not very useful at the moment, it is better to use manual extraction
            #if not use_manual_num:
            #    for dbsf in dbsfile:
            #        inifil=False
            #        ris=os.popen('python $DBSCMD_HOME/dbsCommandLine.py -c search --query="find file.numevents where file like *'+dbsf+'*"')
            #        for lnris in ris:
            #            if inifil:
            #                nevt.append(lnris)
            #            else:
            #                if lnris[:3]=="___":
            #                    inifil=True
            #        nevt.pop()
            #        ris.close()
            #    for nevtn,nevte in nevt:
            #        nevt[nevtn]=int(nevt[nevtn][:-2])
            #        nevent
            #    for nevte in nevt:
            #        

    #for f in fil: remember indentation if uncommenting this
    if not dbs:
        runcmd(environment,"cmsDriver.py","testALCA","-s","ALCA:Configuration/StandardSequences/AlCaRecoStream_EcalCalPhiSym_cff:EcalCalPhiSym+DQM","-n",num_evts,"--filein",fil[nv],"--fileout","file:dqm.root","--eventcontent","FEVT","--conditions","FrontierConditions_GlobalTag,"+conditions+"::All")#,"--no_exec")
    else:
        sfl=""
        for fl in dbsfile:
            sfl=sfl+','+fl
        sfl=sfl[1:]
        runcmd(environment,"cmsDriver.py","testALCA","-s","ALCA:Configuration/StandardSequences/AlCaRecoStream_EcalCalPhiSym_cff:EcalCalPhiSym+DQM","-n",num_evts,"--filein",sfl,"--fileout","file:dqm.root","--eventcontent","FEVT","--conditions","FrontierConditions_GlobalTag,"+conditions+"::All","--no_exec")
        alcareco=open("testALCA_ALCA_"+conditions_file+".py",'r')
        alcarecoln=alcareco.readlines()
        alcareco.close()
        arnum=0
        for arln,arl in enumerate(alcarecoln):         
            if sfl in arl:
                arnum=arln
        alcarecoln[arnum]=alcarecoln[arnum].replace(",","','")
        alcareco=open("testALCA_ALCA_"+conditions_file+".py",'w')
        for arln in alcarecoln:
            alcareco.write(arln)
        alcareco.close()
        runcmd(environment,"cmsRun","testALCA_ALCA_"+conditions_file+".py")
    os.system("mv ALCARECOEcalCalPhiSym.root dqm.root")
    if cvs:
        runcmd(environment,"cmsRun","src/DQMOffline/CalibCalo/test/edm2me_cfg.py")
    else:
        runcmd(environment,"cmsRun",environment["CMSSW_RELEASE_BASE"]+"/src/DQMOffline/CalibCalo/test/edm2me_cfg.py")
    os.system("mv DQM_V0001_R000000001__A__B__C.root "+out_root)
    os.system("rm dqm.root")
    os.chdir("../")
