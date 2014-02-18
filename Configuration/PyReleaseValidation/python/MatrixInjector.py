import sys
import json
import os
import copy
import multiprocessing

def performInjectionOptionTest(opt):
    if opt.show:
        print 'Not injecting to wmagent in --show mode. Need to run the worklfows.'
        sys.exit(-1)
    if opt.wmcontrol=='init':
        #init means it'll be in test mode
        opt.nThreads=0
    if opt.wmcontrol=='test':
        #means the wf were created already, and we just dryRun it.
        opt.dryRun=True
    if opt.wmcontrol=='submit' and opt.nThreads==0:
        print 'Not injecting to wmagent in -j 0 mode. Need to run the worklfows.'
        sys.exit(-1)
    if opt.wmcontrol=='force':
        print "This is an expert setting, you'd better know what you're doing"
        opt.dryRun=True

def upload_to_couch_oneArg(arguments):
    from modules.wma import upload_to_couch
    (filePath,labelInCouch,user,group,where) = arguments
    cacheId=upload_to_couch(filePath,
                            labelInCouch,
                            user,
                            group,
                            test_mode=False,
                            url=where)
    return cacheId


class MatrixInjector(object):

    def __init__(self,opt,mode='init',options=''):
        self.count=1040

        self.dqmgui=None
        self.wmagent=None
        for k in options.split(','):
            if k.startswith('dqm:'):
                self.dqmgui=k.split(':',1)[-1]
            elif k.startswith('wma:'):
                self.wmagent=k.split(':',1)[-1]

        self.testMode=((mode!='submit') and (mode!='force'))
        self.version =1
        self.keep = opt.keep

        #wagemt stuff
        if not self.wmagent:
            self.wmagent=os.getenv('WMAGENT_REQMGR')
        if not self.wmagent:
            self.wmagent = 'cmsweb.cern.ch'

        if not self.dqmgui:
            self.dqmgui="https://cmsweb.cern.ch/dqm/relval"
        #couch stuff
        self.couch = 'https://'+self.wmagent+'/couchdb'
#        self.couchDB = 'reqmgr_config_cache'
        self.couchCache={} # so that we do not upload like crazy, and recyle cfgs
        self.user = os.getenv('USER')
        self.group = 'ppd'
        self.label = 'RelValSet_'+os.getenv('CMSSW_VERSION').replace('-','')+'_v'+str(self.version)
        self.speciallabel=''
        if opt.label:
            self.speciallabel= '_'+opt.label


        if not os.getenv('WMCORE_ROOT'):
            print '\n\twmclient is not setup properly. Will not be able to upload or submit requests.\n'
            if not self.testMode:
                print '\n\t QUIT\n'
                sys.exit(-18)
        else:
            print '\n\tFound wmclient\n'
            
        self.defaultChain={
            "RequestType" :   "TaskChain",                    #this is how we handle relvals
            "Requestor": self.user,                           #Person responsible
            "Group": self.group,                              #group for the request
            "CMSSWVersion": os.getenv('CMSSW_VERSION'),       #CMSSW Version (used for all tasks in chain)
            "Campaign": os.getenv('CMSSW_VERSION'),           # only for wmstat purpose
            "ScramArch": os.getenv('SCRAM_ARCH'),             #Scram Arch (used for all tasks in chain)
            "ProcessingVersion": self.version,                #Processing Version (used for all tasks in chain)
            "GlobalTag": None,                                #Global Tag (overridden per task)
            "CouchURL": self.couch,                           #URL of CouchDB containing Config Cache
            "ConfigCacheURL": self.couch,                     #URL of CouchDB containing Config Cache
            "DbsUrl": "https://cmsweb.cern.ch/dbs/prod/global/DBSReader",
            #- Will contain all configs for all Tasks
            #"SiteWhitelist" : ["T2_CH_CERN", "T1_US_FNAL"],   #Site whitelist
            "TaskChain" : None,                                  #Define number of tasks in chain.
            "nowmTasklist" : [],  #a list of tasks as we put them in
            "unmergedLFNBase" : "/store/unmerged",
            "mergedLFNBase" : "/store/relval",
            "dashboardActivity" : "relval",
            "Memory" : 2400,
            "SizePerEvent" : 1234,
            "TimePerEvent" : 20
            }

        self.defaultHarvest={
            "EnableHarvesting" : "True",
            "DQMUploadUrl" : self.dqmgui,
            "DQMConfigCacheID" : None
            }
        
        self.defaultScratch={
            "TaskName" : None,                            #Task Name
            "ConfigCacheID" : None,                   #Generator Config id
            "GlobalTag": None,
            "SplittingAlgo"  : "EventBased",             #Splitting Algorithm
            "EventsPerJob" : None,                       #Size of jobs in terms of splitting algorithm
            "RequestNumEvents" : None,                      #Total number of events to generate
            "Seeding" : "AutomaticSeeding",                          #Random seeding method
            "PrimaryDataset" : None,                          #Primary Dataset to be created
            "nowmIO": {},
            "KeepOutput" : False
            }
        self.defaultInput={
            "TaskName" : "DigiHLT",                                      #Task Name
            "ConfigCacheID" : None,                                      #Processing Config id
            "GlobalTag": None,
            "InputDataset" : None,                                       #Input Dataset to be processed
            "SplittingAlgo"  : "LumiBased",                        #Splitting Algorithm
            "LumisPerJob" : 10,               #Size of jobs in terms of splitting algorithm
            "nowmIO": {},
            "KeepOutput" : False
            }
        self.defaultTask={
            "TaskName" : None,                                 #Task Name
            "InputTask" : None,                                #Input Task Name (Task Name field of a previous Task entry)
            "InputFromOutputModule" : None,                    #OutputModule name in the input task that will provide files to process
            "ConfigCacheID" : None,                            #Processing Config id
            "GlobalTag": None,
            "SplittingAlgo"  : "LumiBased",                        #Splitting Algorithm
            "LumisPerJob" : 10,               #Size of jobs in terms of splitting algorithm
            "nowmIO": {},
            "KeepOutput" : False
            }

        self.chainDicts={}


    def prepare(self,mReader, directories, mode='init'):
        try:
            #from Configuration.PyReleaseValidation.relval_steps import wmsplit
            wmsplit = {}
            wmsplit['DIGIHI']=5
            wmsplit['RECOHI']=5
            wmsplit['HLTD']=5
            wmsplit['RECODreHLT']=2  
            wmsplit['DIGIPU']=4
            wmsplit['DIGIPU1']=4
            wmsplit['RECOPU1']=1
            wmsplit['DIGIHISt3']=5
            wmsplit['RECOHISt4']=5
            wmsplit['SingleMuPt10_ID']=1
            wmsplit['DIGI_ID']=1
            wmsplit['RECO_ID']=1
            wmsplit['TTbar_ID']=1
            wmsplit['SingleMuPt10FS_ID']=1
            wmsplit['TTbarFS_ID']=1
                                    
            #import pprint
            #pprint.pprint(wmsplit)            
        except:
            print "Not set up for step splitting"
            wmsplit={}

        acqEra=False
        for (n,dir) in directories.items():
            chainDict=copy.deepcopy(self.defaultChain)
            print "inspecting",dir
            nextHasDSInput=None
            for (x,s) in mReader.workFlowSteps.items():
                #x has the format (num, prefix)
                #s has the format (num, name, commands, stepList)
                if x[0]==n:
                    #print "found",n,s[3]
                    #chainDict['RequestString']='RV'+chainDict['CMSSWVersion']+s[1].split('+')[0]
                    index=0
                    splitForThisWf=None
                    thisLabel=self.speciallabel
                    processStrPrefix=''
                    setPrimaryDs=None
                    for step in s[3]:
                        
                        if 'INPUT' in step or (not isinstance(s[2][index],str)):
                            nextHasDSInput=s[2][index]

                        else:

                            if (index==0):
                                #first step and not input -> gen part
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultScratch))
                                try:
                                    chainDict['nowmTasklist'][-1]['nowmIO']=json.loads(open('%s/%s.io'%(dir,step)).read())
                                except:
                                    print "Failed to find",'%s/%s.io'%(dir,step),".The workflows were probably not run on cfg not created"
                                    return -15

                                chainDict['nowmTasklist'][-1]['PrimaryDataset']='RelVal'+s[1].split('+')[0]
                                if not '--relval' in s[2][index]:
                                    print 'Impossible to create task from scratch without splitting information with --relval'
                                    return -12
                                else:
                                    arg=s[2][index].split()
                                    ns=map(int,arg[arg.index('--relval')+1].split(','))
                                    chainDict['nowmTasklist'][-1]['RequestNumEvents'] = ns[0]
                                    chainDict['nowmTasklist'][-1]['EventsPerJob'] = ns[1]
                                if 'FASTSIM' in s[2][index] or '--fast' in s[2][index]:
                                    thisLabel+='_FastSim'

                            elif nextHasDSInput:
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultInput))
                                try:
                                    chainDict['nowmTasklist'][-1]['nowmIO']=json.loads(open('%s/%s.io'%(dir,step)).read())
                                except:
                                    print "Failed to find",'%s/%s.io'%(dir,step),".The workflows were probably not run on cfg not created"
                                    return -15
                                chainDict['nowmTasklist'][-1]['InputDataset']=nextHasDSInput.dataSet
                                splitForThisWf=nextHasDSInput.split
                                chainDict['nowmTasklist'][-1]['LumisPerJob']=splitForThisWf
                                if step in wmsplit:
                                    chainDict['nowmTasklist'][-1]['LumisPerJob']=wmsplit[step]
                                # get the run numbers or #events
                                if len(nextHasDSInput.run):
                                    chainDict['nowmTasklist'][-1]['RunWhitelist']=nextHasDSInput.run
                                #print "what is s",s[2][index]
                                if '--data' in s[2][index] and nextHasDSInput.label:
                                    thisLabel+='_RelVal_%s'%nextHasDSInput.label
                                if 'filter' in chainDict['nowmTasklist'][-1]['nowmIO']:
                                    print "This has an input DS and a filter sequence: very likely to be the PyQuen sample"
                                    processStrPrefix='PU_'
                                    setPrimaryDs = 'RelVal'+s[1].split('+')[0]
                                    if setPrimaryDs:
                                        chainDict['nowmTasklist'][-1]['PrimaryDataset']=setPrimaryDs
                                nextHasDSInput=None
                            else:
                                #not first step and no inputDS
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultTask))
                                try:
                                    chainDict['nowmTasklist'][-1]['nowmIO']=json.loads(open('%s/%s.io'%(dir,step)).read())
                                except:
                                    print "Failed to find",'%s/%s.io'%(dir,step),".The workflows were probably not run on cfg not created"
                                    return -15
                                if splitForThisWf:
                                    chainDict['nowmTasklist'][-1]['LumisPerJob']=splitForThisWf
                                if step in wmsplit:
                                    chainDict['nowmTasklist'][-1]['LumisPerJob']=wmsplit[step]

                            #print step
                            chainDict['nowmTasklist'][-1]['TaskName']=step
                            if setPrimaryDs:
                                chainDict['nowmTasklist'][-1]['PrimaryDataset']=setPrimaryDs
                            chainDict['nowmTasklist'][-1]['ConfigCacheID']='%s/%s.py'%(dir,step)
                            chainDict['nowmTasklist'][-1]['GlobalTag']=chainDict['nowmTasklist'][-1]['nowmIO']['GT'] # copy to the proper parameter name
                            chainDict['GlobalTag']=chainDict['nowmTasklist'][-1]['nowmIO']['GT'] #set in general to the last one of the chain
                            if 'pileup' in chainDict['nowmTasklist'][-1]['nowmIO']:
                                chainDict['nowmTasklist'][-1]['MCPileup']=chainDict['nowmTasklist'][-1]['nowmIO']['pileup']
                            if '--pileup' in s[2][index]:
                                processStrPrefix='PU_'
                                
                            if acqEra:
                                #chainDict['AcquisitionEra'][step]=(chainDict['CMSSWVersion']+'-PU_'+chainDict['nowmTasklist'][-1]['GlobalTag']).replace('::All','')+thisLabel
                                chainDict['AcquisitionEra'][step]=chainDict['CMSSWVersion']
                                chainDict['ProcessingString'][step]=processStrPrefix+chainDict['nowmTasklist'][-1]['GlobalTag'].replace('::All','')+thisLabel
                            else:
                                #chainDict['nowmTasklist'][-1]['AcquisitionEra']=(chainDict['CMSSWVersion']+'-PU_'+chainDict['nowmTasklist'][-1]['GlobalTag']).replace('::All','')+thisLabel
                                chainDict['nowmTasklist'][-1]['AcquisitionEra']=chainDict['CMSSWVersion']
                                chainDict['nowmTasklist'][-1]['ProcessingString']=processStrPrefix+chainDict['nowmTasklist'][-1]['GlobalTag'].replace('::All','')+thisLabel

                        index+=1
                    #end of loop through steps
                    chainDict['RequestString']='RV'+chainDict['CMSSWVersion']+s[1].split('+')[0]
                    if processStrPrefix or thisLabel:
                        chainDict['RequestString']+='_'+processStrPrefix+thisLabel

                        
                        
            #wrap up for this one
            import pprint
            #print 'wrapping up'
            #pprint.pprint(chainDict)
            #loop on the task list
            for i_second in reversed(range(len(chainDict['nowmTasklist']))):
                t_second=chainDict['nowmTasklist'][i_second]
                #print "t_second taskname", t_second['TaskName']
                if 'primary' in t_second['nowmIO']:
                    #print t_second['nowmIO']['primary']
                    primary=t_second['nowmIO']['primary'][0].replace('file:','')
                    for i_input in reversed(range(0,i_second)):
                        t_input=chainDict['nowmTasklist'][i_input]
                        for (om,o) in t_input['nowmIO'].items():
                            if primary in o:
                                #print "found",primary,"procuced by",om,"of",t_input['TaskName']
                                t_second['InputTask'] = t_input['TaskName']
                                t_second['InputFromOutputModule'] = om
                                #print 't_second',pprint.pformat(t_second)
                                if t_second['TaskName'].startswith('HARVEST'):
                                    chainDict.update(copy.deepcopy(self.defaultHarvest))
                                    chainDict['DQMConfigCacheID']=t_second['ConfigCacheID']
                                    ## the info are not in the task specific dict but in the general dict
                                    #t_input.update(copy.deepcopy(self.defaultHarvest))
                                    #t_input['DQMConfigCacheID']=t_second['ConfigCacheID']
                                break

            ## there is in fact only one acquisition era
            #if len(set(chainDict['AcquisitionEra'].values()))==1:
            #    print "setting only one acq"
            if acqEra:
                chainDict['AcquisitionEra'] = chainDict['AcquisitionEra'].values()[0]
                
            ## clean things up now
            itask=0
            if self.keep:
                for i in self.keep:
                    if type(i)==int and i < len(chainDict['nowmTasklist']):
                        chainDict['nowmTasklist'][i]['KeepOutput']=True
            for (i,t) in enumerate(chainDict['nowmTasklist']):
                if t['TaskName'].startswith('HARVEST'):
                    continue
                if not self.keep:
                    t['KeepOutput']=True
                elif t['TaskName'] in self.keep:
                    t['KeepOutput']=True
                t.pop('nowmIO')
                itask+=1
                chainDict['Task%d'%(itask)]=t


            ## 


            ## provide the number of tasks
            chainDict['TaskChain']=itask#len(chainDict['nowmTasklist'])
            
            chainDict.pop('nowmTasklist')
            self.chainDicts[n]=chainDict

            
        return 0

    def uploadConf(self,filePath,label,where):
        labelInCouch=self.label+'_'+label
        cacheName=filePath.split('/')[-1]
        if self.testMode:
            self.count+=1
            print '\tFake upload of',filePath,'to couch with label',labelInCouch
            return self.count
        else:
            try:
                from modules.wma import upload_to_couch,DATABASE_NAME
            except:
                print '\n\tUnable to find wmcontrol modules. Please include it in your python path\n'
                print '\n\t QUIT\n'
                sys.exit(-16)

            if cacheName in self.couchCache:
                print "Not re-uploading",filePath,"to",where,"for",label
                cacheId=self.couchCache[cacheName]
            else:
                print "Loading",filePath,"to",where,"for",label
                ## totally fork the upload to couch to prevent cross loading of process configurations
                pool = multiprocessing.Pool(1)
                cacheIds = pool.map( upload_to_couch_oneArg, [(filePath,labelInCouch,self.user,self.group,where)] )
                cacheId = cacheIds[0]
                self.couchCache[cacheName]=cacheId
            return cacheId
    
    def upload(self):
        for (n,d) in self.chainDicts.items():
            for it in d:
                if it.startswith("Task") and it!='TaskChain':
                    #upload
                    couchID=self.uploadConf(d[it]['ConfigCacheID'],
                                            str(n)+d[it]['TaskName'],
                                            d['CouchURL']
                                            )
                    print d[it]['ConfigCacheID']," uploaded to couchDB for",str(n),"with ID",couchID
                    d[it]['ConfigCacheID']=couchID
                if it =='DQMConfigCacheID':
                    couchID=self.uploadConf(d['DQMConfigCacheID'],
                                            str(n)+'harvesting',
                                            d['CouchURL']
                                            )
                    print d['DQMConfigCacheID'],"uploaded to couchDB for",str(n),"with ID",couchID
                    d['DQMConfigCacheID']=couchID
                        
            
    def submit(self):
        try:
            from modules.wma import makeRequest,approveRequest
            from wmcontrol import random_sleep
            print '\n\tFound wmcontrol\n'
        except:
            print '\n\tUnable to find wmcontrol modules. Please include it in your python path\n'
            if not self.testMode:
                print '\n\t QUIT\n'
                sys.exit(-17)

        import pprint
        for (n,d) in self.chainDicts.items():
            if self.testMode:
                print "Only viewing request",n
                print pprint.pprint(d)
            else:
                #submit to wmagent each dict
                print "For eyes before submitting",n
                print pprint.pprint(d)
                print "Submitting",n,"..........."
                workFlow=makeRequest(self.wmagent,d,encodeDict=True)
                approveRequest(self.wmagent,workFlow)
                print "...........",n,"submitted"
                random_sleep()
            

        
