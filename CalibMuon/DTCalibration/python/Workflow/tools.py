import os,sys,imp
import ConfigParser

def replaceTemplate(template,**opts):
    result = open(template).read()
    for item in opts:
         old = '@@%s@@'%item
         new = str(opts[item])
         print "Replacing",old,"to",new
         result = result.replace(old,new)

    return result
 
def listFilesInCastor(castor_dir,type = 'root'):
    if not castor_dir: raise ValueError,'Please specify valid castor dir'

    from subprocess import Popen,PIPE
    p1 = Popen(['nsls',castor_dir],stdout=PIPE)
    p2 = Popen(['grep',type],stdin=p1.stdout,stdout=PIPE)
    files = ['rfio:' + castor_dir + "/" + item[:-1] for item in p2.stdout]
    p2.stdout.close()
    return files

def listFilesLocal(dir,type = 'root'):
    if not dir: raise ValueError,'Please specify valid dir'

    from subprocess import Popen,PIPE
    p1 = Popen(['ls',dir],stdout=PIPE)
    p2 = Popen(['grep',type],stdin=p1.stdout,stdout=PIPE)
    files = [dir + "/" + item[:-1] for item in p2.stdout]
    p2.stdout.close()
    return files

def haddInCastor(castor_dir,result_file,type = 'root'):
    if not castor_dir: raise ValueError,'Please specify valid castor dir'
    if not result_file: raise ValueError,'Please specify valid output file name'

    #cmd = 'hadd %s `./listfilesCastor %s | grep %s`'%(result_file,castor_dir,type)
    #print "Running",cmd
    #os.system(cmd)
    from subprocess import call
    files = listFilesInCastor(castor_dir,type)
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

def loadCmsProcess(pset_name):
    pset = imp.load_source("psetmodule",pset_name)
    return pset.process

def writeCfg(process,dir,pset_name):
    if not os.path.exists(dir): os.makedirs(dir)
    open(dir + '/' + pset_name,'w').write(process.dumpPython())

def loadCrabCfg(cfg_name):
    config = ConfigParser.ConfigParser()
    config.read(cfg_name)
    return config

def loadCrabDefault(crab_cfg,config):
    # CRAB section
    if not crab_cfg.has_section('CRAB'): crab_cfg.add_section('CRAB')
    crab_cfg.set('CRAB','jobtype','cmssw')
    crab_cfg.set('CRAB','scheduler',config.scheduler) 
    if config.useserver: crab_cfg.set('CRAB','use_server',1)
    # CMSSW section
    if not crab_cfg.has_section('CMSSW'): crab_cfg.add_section('CMSSW')
    crab_cfg.set('CMSSW','datasetpath',config.datasetpath)
    crab_cfg.set('CMSSW','pset','pset.py')
    # Splitting config
    crab_cfg.set('CMSSW','runselection',config.runselection) 
    if hasattr(config,'totalnumberevents'): crab_cfg.set('CMSSW','total_number_of_events',config.totalnumberevents)
    if hasattr(config,'eventsperjob'): crab_cfg.set('CMSSW','events_per_job',config.eventsperjob)
    # USER section
    if not crab_cfg.has_section('USER'): crab_cfg.add_section('USER')  
    # Stageout config
    if hasattr(config,'stageOutCAF') and config.stageOutCAF:
        crab_cfg.set('USER','return_data',0)                
        crab_cfg.set('USER','copy_data',1)  
        crab_cfg.set('USER','storage_element','T2_CH_CAF')
        crab_cfg.set('USER','user_remote_dir',config.userdircaf)
        crab_cfg.set('USER','check_user_remote_dir',0)
    elif hasattr(config,'stageOutLocal') and config.stageOutLocal:
        crab_cfg.set('USER','return_data',1)                
        crab_cfg.set('USER','copy_data',0)
        crab_cfg.remove_option('USER','storage_element')
        crab_cfg.remove_option('USER','user_remote_dir')
        crab_cfg.remove_option('USER','check_user_remote_dir')

    if hasattr(config,'email') and config.email: crab_cfg.set('USER','eMail',config.email)
    crab_cfg.set('USER','xml_report','crabReport.xml')

    if hasattr(config,'runOnGrid') and config.runOnGrid:
        crab_cfg.remove_section('CAF')
