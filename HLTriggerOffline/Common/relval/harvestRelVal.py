#!/usr/bin/env python

import sys
import os

"""
arguments [<list-of-processes>]
description:
creates crab.cfg, multicrab.cfg, harvest_*.py
if dbs is set:
 prints number of events found in dataset
 if no argument is provided looks for all available datsets for release
 user can edit multicrab and confirm process list as needed
nuno@cern.ch 09.04
"""

def print_def():
    print "Usage:", sys.argv[0], "[list_of_processes]"
    print "Examples:"
    print "harvestRelVal.py"
    print "harvestRelVal.py /RelValTTbar/CMSSW_3_1_0_pre4_STARTUP_30X_v1/GEN-SIM-RECO"
    print "harvestRelVal.py <dataset_list.txt>"

def check_dbs():
    if os.getenv('DBSCMD_HOME','NOTSET') == 'NOTSET' :
        return 0
    return 1

def check_nevts_dset(dset):
    if not is_dbs :
        return -1
    ntot=0
    for afile in api.listFiles(path=str(dset)):
        nevts = afile['NumberOfEvents']
        ntot += nevts
        #print "  %s" % afile['LogicalFileName']
    return ntot  

def make_dqmname(s):
    return  'DQM_V0001_R000000001' + s.replace('/','__') + '.root' 

def get_name_from_dsetpath(ds):
    fs = ds.split('/')
    fa = fs[1].replace('RelVal','')
    return fa

def get_cond_from_dsetpath(ds) :
    ca = ds.split('/')[2].replace(cmssw_ver+'_','').replace('IDEAL_','').replace('STARTUP_','').replace('_FastSim','')
    cb = ca[:ca.find('v')-1]
    if cb[0].find('3') == -1 or len(cb) > 3:
        print "problem extracting condition for", ds, " : ", cb, '(len:',len(cb),')'  
        if cb.find('31X') != -1:
            cb = '31X'
        elif cb.find('30X') != -1:
            cb = '30X'
        else:
            print "skipping", cb
            return 0
        print "condition found:", cb
    else :
        print "good condition for", ds, " : ", cb, '(len:',len(cb),')'      
    return cb


def make_dbs_list(dbslf) :
    if not is_dbs :
        return
    flis = open(dbslf,'w')
    for ads in api.listDatasetPaths() :
        if ads.find('RelVal') != -1 \
               or ads.find(cmssw_ver) != -1 \
               or ads.find("/GEN-SIM") != -1 : 
#               and ads.find("/GEN-SIM-RECO") != -1 : 
            flis.write(ads + '\n')
    flis.close()
    print 'Generated dataset list', dbslf, 'from dbs.' 
    #exampe:
    #dbs lsd --path=/RelVal*/CMSSW_3_1_0_pre5*/GEN-SIM-RECO --url=http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet > mylist.txt
    #dbslsd = "dbs lsd --path=/RelVal*/" + cmssw_ver + "*/GEN-SIM-RECO --url=http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet"
    #os.system( '`' + dbslsd + ' > ' + dbslf + '`')

def read_ds_file() :
    if not os.path.exists(dsfile) :
        print "problem reading file", dsfile
        sys.exit(30)
    fin = open(dsfile,'r')
    for dset in fin.readlines(): 
        d = dset.replace('\n','')
        if d.find('#') == -1 :
            dsetpaths.append(d)
        else :
            print 'skipping:', d
    fin.close()
    print 'Using data set list in ', dsfile

def check_dset() :
   #check cmssw consistency
   for s in dsetpaths:
       if s.find(cmssw_ver) == -1 :
           dsetpaths.remove(s)        
           print 'Inconsistency found with datset and cmssw version (', cmssw_ver, ')' \
                 ': \t ', s, ' has been removed.'
   #check conditions from dsetname
   for s in dsetpaths[:]: #nb:need to make a copy here!
       cond = get_cond_from_dsetpath(s)
       if cond  == 0 : 
           dsetpaths.remove(s)        
   #check list size
   nSamples = len(dsetpaths)
   if nSamples == 0 :
       print "Empty input list, exit."
       sys.exit(12)
   else :
       print 'Processing', nSamples, 'data sets.'
   #check event numbers
   nSampleEvts = list()
   for s in dsetpaths:
       nSampleEvts.append(check_nevts_dset(s))
   print 'number of events per dataset:', nSampleEvts

def find_dqmref(ds) :
    if not do_reference :
        return 'NONE'
    cp = cmssw_ver[-1:]
    ip = (int)(cp) - 1
    ref_ver = cmssw_ver.replace(cp,str(ip))
    #print "cms:", cmssw_ver, " cp:", cp, " ip:", ip, " new_ver:", ref_ver  
    ref_dir = "/castor/cern.ch/user/n/nuno/relval/harvest/" + ref_ver + "/"
    ref_dsf = make_dqmname(ds.replace(cmssw_ver, ref_ver))
    gls = " | grep root | grep "
    #to accept crab appended _1.root in file names, nd skip versions/conditions
    gls += ref_dsf[:-25] 
    gls += "| awk '{print $9}' "
    #print "refds:", ref_dsf, " command: rfdir", ref_dir+gls
    command = "rfcp " + ref_dir  + "`rfdir " + ref_dir + gls + "` ."
    #print "command:", command
    os.system(command)
    tmpfile = "ref.txt"
    command = "ls -rtl *" + gls + " > " + tmpfile
    #print "command:", command
    os.system(command)
    the_ref = 'NONE'
    if os.path.exists(tmpfile) :
        fin = open(tmpfile,'r')
        ref = fin.readline().replace('\n','')
        #print "read ref:", ref, "exists?", os.path.exists(ref)
        fin.close()
        if os.path.exists(ref) :
            the_ref = ref
    else :
        the_ref = 'NONE'
    print "Found reference file:", the_ref
    return the_ref

def create_harvest(ds) :
    raw_cmsdriver = "cmsDriver.py harvest -s HARVESTING:validationHarvesting --mc  --conditions FrontierConditions_GlobalTag,STARTUP_30X::All --harvesting AtJobEnd --no_exec -n -1"
    cmsdriver = raw_cmsdriver
    cond = get_cond_from_dsetpath(ds)
    if cond == 0 :
        print 'unexpected problem with conditions'
        sys.exit(50)
    cmsdriver = cmsdriver.replace('30X',cond)
    fin_name="harvest_HARVESTING_STARTUP.py"
    if ds.find('IDEAL') != -1 :
        cmsdriver = cmsdriver.replace('STARTUP','IDEAL')
        fin_name = fin_name.replace('STARTUP','IDEAL')
    if ds.find('FastSim') != -1:
        cmsdriver = cmsdriver.replace('validationHarvesting','validationHarvestingFS')
    if ds.find('PileUp') != -1:
        cmsdriver = cmsdriver.replace('validationHarvesting','validationHarvestingPU')

    #print "=>", cmsdriver, " fs?", ds.find('FastSim')
    if os.path.exists(fin_name) : 
        os.system("rm " + fin_name)
    print "executing cmsdriver command:\n\t", cmsdriver
    os.system(cmsdriver)
    if not os.path.exists(fin_name) : 
        print 'problem with cmsdriver file name'
        sys.exit(40)
    os.system("touch " + fin_name)
    hf = make_harv_name(ds)
    os.system('mv ' + fin_name + " " + hf)
    out = open(hf, 'a')
    out.write("\n\n##additions to cmsDriver output \n")
    out.write("process.dqmSaver.workflow = '" + ds + "'\n")
    if is_dbs :
        out.write("process.source.fileNames = cms.untracked.vstring(\n")
        for afile in api.listFiles(path=ds):
            out.write("  '%s',\n" % afile['LogicalFileName'])
        out.write(")\n")

    dqmref = find_dqmref(ds);
    if not dqmref == 'NONE' : 
        out.write("process.DQMStore.referenceFileName = '" + dqmref + "'\n")
        out.write("process.dqmSaver.referenceHandling = 'all'\n")

    out.close()

def create_mcrab(set, fcrab, fout):
    out = open(fout, 'w')
    out.write('[MULTICRAB]')
    out.write('\ncfg=' + fcrab)
    out.write('\n\n[COMMON]')
    nevt = -1
    njob = 1
    out.write('\nCMSSW.total_number_of_events=' + (str)(nevt) )
    out.write('\nCMSSW.number_of_jobs=' + (str)(njob) )
    for s in set:
        append_sample_mcrab(s, out)
    out.close()    

def make_harv_name(dset) :
    return 'harvest_' + get_name_from_dsetpath(dset) + '.py' 

def append_sample_mcrab(dsetp, fout):
    dqm = make_dqmname(dsetp)
    sample = get_name_from_dsetpath(dsetp)
    hf = make_harv_name(dsetp)
    if not os.path.exists(hf) :
        print 'problem creating multicrab, file', hf, 'does not exist'
        sys.exit(17)
    fout.write('\n\n[' + sample + ']')
    fout.write('\nCMSSW.pset=' + hf)
    fout.write('\nCMSSW.datasetpath=' + dsetp)
    fout.write('\nCMSSW.output_file=' + dqm)

    dqmref = find_dqmref(dsetp);
    if not dqmref == 'NONE' : 
        fout.write('\nUSER.additional_input_files=' + dqmref)

def create_crab(ds) :
    dqmout = make_dqmname(ds)
    hf = make_harv_name(ds)
    out = open(f_crab, 'w')
    out.write(crab_block)
    out.write('\npset=' + hf)
    out.write('datasetpath=' + ds)
    out.write('\noutput_file=' + dqmout)
    out.close()

crab_block = """
[CRAB]
jobtype = cmssw
scheduler = glite

[EDG]
remove_default_blacklist=1
rb = CERN

[USER]
return_data = 1
#copy_data = 1
#storage_element=srm-cms.cern.ch
#storage_path=/srm/managerv2?SFN=/castor/cern.ch
#user_remote_dir=/user/n/nuno/test
publish_data=0
thresholdLevel=70
eMail=nuno@cern.ch

[CMSSW]
total_number_of_events=-1
show_prod = 1
number_of_jobs=1
"""


#Check arg,settings
input_type = ''
argin = ''
dsfile = ''
do_reference = False
if len(sys.argv) > 2 : 
    print_def()
    sys.exit(10) 
elif len(sys.argv) == 1 : 
    print "Will search for available datasets."
    input_type = 'none'
elif len(sys.argv) == 2 : 
    argin = sys.argv[1]
    if os.path.exists(argin) :
        dsfile = argin
        #print 'Reading list of datasets from', dsfile
        input_type = 'file'
    elif argin.find('CMSSW') != -1 and argin.find('RelVal'): 
        print 'Using specified data set', argin
        input_type = 'ds'
    else :
        print 'Invalid argument: process list, dataset or file', \
                  argin, 'does not exist.'
        sys.exit(11) 

#dbs
is_dbs = check_dbs()
if not is_dbs:
    print "dbs not set!"
else:
    print "dbs home:", os.getenv('DBSCMD_HOME')
    from DBSAPI.dbsApi import DbsApi
    from DBSAPI.dbsException import *
    from DBSAPI.dbsApiException import *
    from DBSAPI.dbsOptions import DbsOptionParser
    optManager  = DbsOptionParser()
    (opts,args) = optManager.getOpt()
    #api = DbsApi(opts.__dict__)
    args={}
    args['url']= "http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet"
    api = DbsApi(args)

#cmssw
cmssw_ver = os.getenv('CMSSW_VERSION','NOTSET')
if cmssw_ver == 'NOTSET' :
    print """
    cmssw not set!
    example:
      scramv1 p CMSSW CMSSW_3_1_0_pre5
      cd CMSSW_3_1_0_pre5/src
      eval `scramv1 runtime -sh`
      cd -
    """
    sys.exit(12) 
else :
    print "Using cmssw version:", cmssw_ver
    

#read datasets
dsetpaths = list()

if input_type == 'none' :
    if not is_dbs :
        print "no dataset specified, and dbs isn't set..."
        print_def()
        sys.exit(13)
    else :
        dsfile = cmssw_ver + "_dbslist.txt"
        make_dbs_list(dsfile)
        read_ds_file()
elif input_type == 'file' :
    read_ds_file()
elif input_type == 'ds' :
    dsetpaths.append(argin)


#check dataset list: remove incompatible dsets
check_dset()

#print dataset list to be processed
print 'data sets:', dsetpaths
dslproc = open("dset_processed.txt", 'w')
for s in dsetpaths :
    dslproc.write(s+'\n')
dslproc.close()


##Create harvest.py template
create_harvest(dsetpaths[0])

##Create crab.cfg template
f_crab = 'crab.cfg'
create_crab(dsetpaths[0])

##Create harvest_n.py for individual datasets
for s in dsetpaths:
    create_harvest(s)

##Create multicrab.cfg
f_multi_crab = 'multicrab.cfg'
create_mcrab(dsetpaths, f_crab, f_multi_crab)

##Print what has been created

harvfilelist = list()
for s in dsetpaths:
    harvfilelist.append(make_harv_name(s))

print '\nCreated:\n\t %(pwd)s/%(cf)s \n\t %(pwd)s/%(mc)s' \
      % {'pwd' : os.environ["PWD"],'cf' : f_crab, 'mc' : f_multi_crab}
print "\tIndividual harvest py's:\n\t", harvfilelist

print "Done."
