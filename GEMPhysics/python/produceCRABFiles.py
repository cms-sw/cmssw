import os

inputFiles = ['WHTolnu2mu_M125GeV_14TeV_cff',
              'WHTo3munu_M125GeV_14TeV_cff',
              'WHTolnu2tau_M125GeV_14TeV_cff',
              'ZHTo2l2mu_M125GeV_14TeV_cff',
              'HToZZ4l_M125GeV_14TeV_cff',
              'muStarTomuZ3mu_M2500GeV_14TeV_cff',
              'muStarTomuZ3mu_M4000GeV_14TeV_cff',
              'WprimeTomunu_M3000GeV_14TeV_cff',
              'WprimeTomunu_M6000GeV_14TeV_cff',
              'ZprimeTo2mu_M3000GeV_14TeV_cff',
              'ZprimeTo2mu_M6000GeV_14TeV_cff',
              'ZTo2mu_14TeV_cff']

events = [100000,100000,100000,100000,200000,50000,50000,50000,50000,50000,50000,500000]
extra = 0.1 ## 10% extra for quick finish

for i in range(len(inputFiles)):
    inputFile = inputFiles[i]
    crabFileName = 'crab.' + inputFile.replace('_cff_GEN_SIM.py','_GEN_SIM') + '.cfg'
    f = open('%s'%(crabFileName), 'w\n')
    f.write('[CMSSW]\n')
    f.write('pset=/uscms_data/d3/dildick/work/CMSSW_6_1_2_SLHC4/src/RPCGEM/GEMPhysics/python/%s\n'%(inputFile))
    event = events[i] * (1+extra)
    f.write('number_of_jobs=%d\n'%(event/1000)) 
    f.write('output_file=out_sim.root\n')
    f.write('total_number_of_events=%d\n'%(event))
    f.write('datasetpath=none\n\n')
    f.write('[GRID]\n')
    f.write('retry_count=0\n')
    f.write('proxy_server=fg-myproxy.fnal.gov\n')
    f.write('virtual_organization=cms\n')
    f.write('rb=CERN\n\n')
    f.write('[USER]\n')
    f.write('srm_version=srmv2\n')
    f.write('ui_working_dir=%s\n'%(inputFile.replace('_cff_GEN_SIM.py','_GEN_SIM')))
    f.write('copy_data=1\n')
    f.write('publish_data_name=%s\n'%(inputFile.replace('_cff_GEN_SIM.py','_GEN_SIM')))
    f.write('publish_data=1\n')
    f.write('check_user_remote_dir=0\n')
    f.write('dbs_url_for_publication=https://cmsdbsprod.cern.ch:8443/cms_dbs_ph_analysis_01_writer/servlet/DBSServlet\n')
    f.write('storage_element=cmssrm.fnal.gov\n')
    f.write('return_data=0\n')
    f.write('user_remote_dir=/store/user/lpcgem/gemphysics/\n')
    f.write('storage_path=/srm/managerv2?SFN=/11\n\n')
    f.write('[CRAB]\n')
    f.write('cfg=crab.cfg\n')
    f.write('scheduler=remoteGlidein\n')
    f.write('jobtype=cmssw\n')
    f.write('use_server = 0\n')
    f.close()

    ## submit crab jobs to the grid
    crabFileName = inputFile.replace('_cff_GEN_SIM.py','') 
    ##os.system('crab -cfg %s -create -submit'%(crabFileName) )
    print 'crab -cfg %s -create -submit'%(crabFileName)

    
    



