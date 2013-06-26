import os,sys,imp
import pickle
import ConfigParser

def replaceTemplate(template,**opts):
    result = open(template).read()
    for item in opts:
         old = '@@%s@@'%item
         new = str(opts[item])
         print "Replacing",old,"to",new
         result = result.replace(old,new)

    return result
 
def getDatasetStr(datasetpath):
    datasetstr = datasetpath
    datasetstr.strip()
    if datasetstr[0] == '/': datasetstr = datasetstr[1:]
    datasetstr = datasetstr.replace('/','_')

    return datasetstr

def dqmWorkflowName(datasetpath,type,rev=1):
    workflowName = datasetpath
    sections = workflowName.split('/')[1:]
    workflowName = '/%s/%s-%s-rev%d/%s' % (sections[0],sections[1],type,rev,sections[2])
    
    return workflowName 
   
def listFilesInCastor(castor_dir,type = 'root',prefix = 'rfio:'):
    if not castor_dir: raise ValueError,'Please specify valid castor dir'

    from subprocess import Popen,PIPE
    p1 = Popen(['nsls',castor_dir],stdout=PIPE)
    #p2 = Popen(['grep',type],stdin=p1.stdout,stdout=PIPE)
    #files = [prefix + castor_dir + "/" + item[:-1] for item in p2.stdout]
    #p2.stdout.close()
    files = [ "%s%s/%s" % (prefix,castor_dir,item.rstrip()) for item in p1.stdout if item.find(type) != -1 ] 
    p1.stdout.close()
    return files

def listFilesLocal(dir,type = 'root'):
    if not dir: raise ValueError,'Please specify valid dir'

    #from subprocess import Popen,PIPE
    #p1 = Popen(['ls',dir],stdout=PIPE)
    #p2 = Popen(['grep',type],stdin=p1.stdout,stdout=PIPE)
    #files = [dir + "/" + item[:-1] for item in p2.stdout]
    #p2.stdout.close()
    files = os.listdir(dir)
    files = [ "%s/%s" % (dir,item) for item in files if item.find(type) != -1 ]

    return files

def copyFilesFromCastor(castor_dir,output_dir,type='root'):
    from subprocess import call
    files = listFilesInCastor(castor_dir,type,'')

    print "Copying files from %s to %s" % (castor_dir,output_dir) 
    for item in files:
        cmd = ['rfcp',item,output_dir] 
        print "..." + item
        retcode = call(cmd)
        if retcode != 0: raise RuntimeError,'Error in copying file %s to directory %s' % (item,output_dir)

    return 0

def copyFilesLocal(dir,output_dir,type='root'):
    if not dir: raise ValueError,'Please specify valid dir'
    if not output_dir: raise ValueError,'Please specify valid output dir'
  
    from subprocess import call
    files = listFilesLocal(dir,type)
    cmd = ['cp']
    cmd.extend(files)
    cmd.append(output_dir)
    print cmd 
    retcode = call(cmd)
    return retcode

def haddInCastor(castor_dir,result_file,type = 'root',prefix = 'rfio:',suffix = None):
    if not castor_dir: raise ValueError,'Please specify valid castor dir'
    if not result_file: raise ValueError,'Please specify valid output file name'

    #cmd = 'hadd %s `./listfilesCastor %s | grep %s`'%(result_file,castor_dir,type)
    #print "Running",cmd
    #os.system(cmd)
    from subprocess import call
    files = listFilesInCastor(castor_dir,type,prefix)
    if suffix: files = [item + suffix for item in files]
 
    cmd = ['hadd',result_file]
    cmd.extend(files)
    #print cmd
    retcode = call(cmd)
    return retcode

def haddLocal(dir,result_file,type = 'root'):
    if not dir: raise ValueError,'Please specify valid dir'
    if not result_file: raise ValueError,'Please specify valid output file name'

    from subprocess import call
    files = listFilesLocal(dir,type)
    cmd = ['hadd',result_file]
    cmd.extend(files)
    #print cmd
    retcode = call(cmd)
    return retcode

def setGridEnv(cmssw_dir):
    cwd = os.getcwd()
    os.chdir(cmssw_dir)

    os.system('source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh')
    os.system('cmsenv')
    os.system('source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh')
 
    os.chdir(cwd)
 
    return

def parseInput(inputFields,requiredFields = ()):

    class options: pass
    for item in sys.argv:
        option = item.split('=')[0]
        if option in inputFields:
            value = item.split('=')[1]
            if value in ('true','True','yes','Yes'): value = True
            elif value in ('false','False','no','No'): value = False

            setattr(options,option,value)

    for item in requiredFields:
        if not hasattr(options,item):
            raise RuntimeError,'Need to set "%s"' % item

    return options

def loadCmsProcessFile(psetName):
    pset = imp.load_source("psetmodule",psetName)
    return pset.process

def loadCmsProcess(psetPath):
    module = __import__(psetPath)
    process = sys.modules[psetPath].process

    import copy 
    #FIXME: clone process
    #processNew = copy.deepcopy(process)
    processNew = copy.copy(process) 
    return processNew

def prependPaths(process,seqname):
    for path in process.paths: 
        getattr(process,path)._seq = getattr(process,seqname)*getattr(process,path)._seq

def writeCfg(process,dir,psetName):
    if not os.path.exists(dir): os.makedirs(dir)
    open(dir + '/' + psetName,'w').write(process.dumpPython())

def writeCfgPkl(process,dir,psetName):
    if not os.path.exists(dir): os.makedirs(dir)

    pklFileName = psetName.split('.')[0] + '.pkl'
    pklFile = open(dir + '/' + pklFileName,"wb")
    myPickle = pickle.Pickler(pklFile)
    myPickle.dump(process)
    pklFile.close()
 
    outFile = open(dir + '/' + psetName,"w")
    outFile.write("import FWCore.ParameterSet.Config as cms\n")
    outFile.write("import pickle\n")
    outFile.write("process = pickle.load(open('%s', 'rb'))\n" % pklFileName)
    outFile.close()


def loadCrabCfg(cfgName=None):
    config = ConfigParser.ConfigParser()
    if cfgName: config.read(cfgName)
    return config

def addCrabInputFile(crabCfg,inputFile):
    additionalInputFiles = ''
    if crabCfg.has_option('USER','additional_input_files'):
        additionalInputFiles = crabCfg.get('USER','additional_input_files')

    if additionalInputFiles: additionalInputFiles += ',%s' % inputFile
    else: additionalInputFiles = inputFile

    crabCfg.set('USER','additional_input_files',additionalInputFiles)

    return crabCfg

def loadCrabDefault(crabCfg,config):
    # CRAB section
    if not crabCfg.has_section('CRAB'): crabCfg.add_section('CRAB')
    crabCfg.set('CRAB','jobtype','cmssw')

    if hasattr(config,'scheduler') and config.scheduler: crabCfg.set('CRAB','scheduler',config.scheduler) 
    else: crabCfg.set('CRAB','scheduler','CAF')

    if hasattr(config,'useserver') and config.useserver: crabCfg.set('CRAB','use_server',1)

    # CMSSW section
    if not crabCfg.has_section('CMSSW'): crabCfg.add_section('CMSSW')
    if hasattr(config,'datasetpath') and config.datasetpath: crabCfg.set('CMSSW','datasetpath',config.datasetpath)
    else: crabCfg.set('CMSSW','datasetpath','/XXX/YYY/ZZZ') 
    crabCfg.set('CMSSW','pset','pset.py')

    # Splitting config
    crabCfg.remove_option('CMSSW','total_number_of_events')
    crabCfg.remove_option('CMSSW','events_per_job')
    crabCfg.remove_option('CMSSW','number_of_jobs')
    crabCfg.remove_option('CMSSW','total_number_of_lumis')
    crabCfg.remove_option('CMSSW','lumis_per_job')
    crabCfg.remove_option('CMSSW','lumi_mask')
    crabCfg.remove_option('CMSSW','split_by_run')
 
    """
    if hasattr(config,'totalnumberevents'): crabCfg.set('CMSSW','total_number_of_events',config.totalnumberevents)
    if hasattr(config,'eventsperjob'): crabCfg.set('CMSSW','events_per_job',config.eventsperjob) 
    """
    if hasattr(config,'splitByLumi') and config.splitByLumi:
        crabCfg.set('CMSSW','total_number_of_lumis',config.totalnumberlumis)
        crabCfg.set('CMSSW','lumis_per_job',config.lumisperjob)
        if hasattr(config,'lumimask') and config.lumimask: crabCfg.set('CMSSW','lumi_mask',config.lumimask)
    elif hasattr(config,'splitByEvent') and config.splitByEvent:
        crabCfg.set('CMSSW','total_number_of_events',config.totalnumberevents)
        crabCfg.set('CMSSW','events_per_job',config.eventsperjob)
    else:
        crabCfg.set('CMSSW','split_by_run',1)

    if hasattr(config,'splitByEvent') and config.splitByEvent:
        crabCfg.remove_option('CMSSW','runselection')
    else:
	if hasattr(config,'runselection') and config.runselection:
	    crabCfg.set('CMSSW','runselection',config.runselection)

    # USER section
    if not crabCfg.has_section('USER'): crabCfg.add_section('USER')  

    # Stageout config
    if hasattr(config,'stageOutCAF') and config.stageOutCAF:
        crabCfg.set('USER','return_data',0)                
        crabCfg.set('USER','copy_data',1)  
        crabCfg.set('USER','storage_element','T2_CH_CAF')
        crabCfg.set('USER','user_remote_dir',config.userdircaf)
        crabCfg.set('USER','check_user_remote_dir',0)
    elif hasattr(config,'stageOutLocal') and config.stageOutLocal:
        crabCfg.set('USER','return_data',1)                
        crabCfg.set('USER','copy_data',0)
        crabCfg.remove_option('USER','storage_element')
        crabCfg.remove_option('USER','user_remote_dir')
        crabCfg.remove_option('USER','check_user_remote_dir')

    if hasattr(config,'email') and config.email: crabCfg.set('USER','eMail',config.email)
    crabCfg.set('USER','xml_report','crabReport.xml')

    if hasattr(config,'runOnGrid') and config.runOnGrid:
        crabCfg.remove_section('CAF')
        if hasattr(config,'ce_black_list'):
            if not crabCfg.has_section('GRID'): crabCfg.add_section('GRID')
            crabCfg.set('GRID','ce_black_list', config.ce_black_list)
        if hasattr(config,'ce_white_list'):
            if not crabCfg.has_section('GRID'): crabCfg.add_section('GRID')
            crabCfg.set('GRID','ce_white_list', config.ce_white_list)
    else:
        if not crabCfg.has_section('CAF'): crabCfg.add_section('CAF')
        crabCfg.set('CAF','queue',config.queueAtCAF) 
    
    return crabCfg
