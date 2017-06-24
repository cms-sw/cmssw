import sys
import json
import os
import copy
import multiprocessing
import time

def performInjectionOptionTest(opt):
    if opt.show:
        print 'Not injecting to wmagent in --show mode. Need to run the worklfows.'
        sys.exit(-1)
    if opt.wmcontrol=='init':
        #init means it'll be in test mode
        opt.nProcs=0
    if opt.wmcontrol=='test':
        #means the wf were created already, and we just dryRun it.
        opt.dryRun=True
    if opt.wmcontrol=='submit' and opt.nProcs==0:
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
        self.memoryOffset = opt.memoryOffset
        self.memPerCore = opt.memPerCore
        self.batchName = ''
        self.batchTime = str(int(time.time()))
        if(opt.batchName):
            self.batchName = '__'+opt.batchName+'-'+self.batchTime

        #wagemt stuff
        if not self.wmagent:
            self.wmagent=os.getenv('WMAGENT_REQMGR')
        if not self.wmagent:
            if not opt.testbed :
                self.wmagent = 'cmsweb.cern.ch'
                self.DbsUrl = "https://"+self.wmagent+"/dbs/prod/global/DBSReader"
            else :
                self.wmagent = 'cmsweb-testbed.cern.ch'
                self.DbsUrl = "https://"+self.wmagent+"/dbs/int/global/DBSReader"

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
            "RequestType" :    "TaskChain",                    #this is how we handle relvals
            "SubRequestType" : "RelVal",                       #this is how we handle relvals, now that TaskChain is also used for central MC production
            "RequestPriority": 500000,
            "Requestor": self.user,                           #Person responsible
            "Group": self.group,                              #group for the request
            "CMSSWVersion": os.getenv('CMSSW_VERSION'),       #CMSSW Version (used for all tasks in chain)
            "Campaign": os.getenv('CMSSW_VERSION'),           # = AcquisitionEra, will be reset later to the one of first task, will both be the CMSSW_VERSION
            "ScramArch": os.getenv('SCRAM_ARCH'),             #Scram Arch (used for all tasks in chain)
            "ProcessingVersion": self.version,                #Processing Version (used for all tasks in chain)
            "GlobalTag": None,                                #Global Tag (overridden per task)
            "ConfigCacheUrl": self.couch,                     #URL of CouchDB containing Config Cache
            "DbsUrl": self.DbsUrl,
            #- Will contain all configs for all Tasks
            #"SiteWhitelist" : ["T2_CH_CERN", "T1_US_FNAL"],   #Site whitelist
            "TaskChain" : None,                                  #Define number of tasks in chain.
            "nowmTasklist" : [],  #a list of tasks as we put them in
            "Multicore" : 1,   # do not set multicore for the whole chain
            "Memory" : 3000,
            "SizePerEvent" : 1234,
            "TimePerEvent" : 0.1,
            "PrepID": os.getenv('CMSSW_VERSION')
            }

        self.defaultHarvest={
            "EnableHarvesting" : "True",
            "DQMUploadUrl" : self.dqmgui,
            "DQMConfigCacheID" : None,
            "Multicore" : 1              # hardcode Multicore to be 1 for Harvest
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
            "Multicore" : opt.nThreads,                  # this is the per-taskchain Multicore; it's the default assigned to a task if it has no value specified 
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
            "Multicore" : opt.nThreads,                       # this is the per-taskchain Multicore; it's the default assigned to a task if it has no value specified 
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
            "Multicore" : opt.nThreads,                       # this is the per-taskchain Multicore; it's the default assigned to a task if it has no value specified 
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
            wmsplit['DIGIUP15_PU50']=1
            wmsplit['RECOUP15_PU50']=1
            wmsplit['DIGIUP15_PU25']=1
            wmsplit['RECOUP15_PU25']=1
            wmsplit['DIGIUP15_PU25HS']=1
            wmsplit['RECOUP15_PU25HS']=1
            wmsplit['DIGIHIMIX']=5
            wmsplit['RECOHIMIX']=5
            wmsplit['RECODSplit']=1
            wmsplit['SingleMuPt10_UP15_ID']=1
            wmsplit['DIGIUP15_ID']=1
            wmsplit['RECOUP15_ID']=1
            wmsplit['TTbar_13_ID']=1
            wmsplit['SingleMuPt10FS_ID']=1
            wmsplit['TTbarFS_ID']=1
            wmsplit['RECODR2_50nsreHLT']=5
            wmsplit['RECODR2_25nsreHLT']=5
            wmsplit['RECODR2_2016reHLT']=5
            wmsplit['RECODR2_50nsreHLT_HIPM']=5
            wmsplit['RECODR2_25nsreHLT_HIPM']=5
            wmsplit['RECODR2_2016reHLT_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimSingleMu']=5
            wmsplit['RECODR2_2016reHLT_skimDoubleEG']=5
            wmsplit['RECODR2_2016reHLT_skimMuonEG']=5
            wmsplit['RECODR2_2016reHLT_skimJetHT']=5
            wmsplit['RECODR2_2016reHLT_skimMET']=5
            wmsplit['RECODR2_2016reHLT_skimSinglePh']=5
            wmsplit['RECODR2_2016reHLT_skimMuOnia']=5
            wmsplit['RECODR2_2016reHLT_skimSingleMu_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimDoubleEG_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimMuonEG_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimJetHT_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimMET_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimSinglePh_HIPM']=5
            wmsplit['RECODR2_2016reHLT_skimMuOnia_HIPM']=5
            wmsplit['HLTDR2_50ns']=1
            wmsplit['HLTDR2_25ns']=1
            wmsplit['HLTDR2_2016']=1
            wmsplit['Hadronizer']=1
            wmsplit['DIGIUP15']=1 
            wmsplit['RECOUP15']=1 
            wmsplit['RECOAODUP15']=5
            wmsplit['DBLMINIAODMCUP15NODQM']=5
            wmsplit['DigiFull']=5
            wmsplit['RecoFull']=5
            wmsplit['DigiFullPU']=1
            wmsplit['RecoFullPU']=1

                                    
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
                    #if 'HARVESTGEN' in s[3]:
                    if len( [step for step in s[3] if "HARVESTGEN" in step] )>0:
                        chainDict['TimePerEvent']=0.01
                        thisLabel=thisLabel+"_gen"
                    # for double miniAOD test
                    if len( [step for step in s[3] if "DBLMINIAODMCUP15NODQM" in step] )>0:
                        thisLabel=thisLabel+"_dblMiniAOD"
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
                                if 'lhe' in s[2][index] in s[2][index]:
                                    chainDict['nowmTasklist'][-1]['LheInputFiles'] =True

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
                                if len(nextHasDSInput.ls):
                                    chainDict['nowmTasklist'][-1]['LumiList']=nextHasDSInput.ls
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

                            # change LumisPerJob for Hadronizer steps. 
                            if 'Hadronizer' in step: 
                                chainDict['nowmTasklist'][-1]['LumisPerJob']=wmsplit['Hadronizer']

                            #print step
                            chainDict['nowmTasklist'][-1]['TaskName']=step
                            if setPrimaryDs:
                                chainDict['nowmTasklist'][-1]['PrimaryDataset']=setPrimaryDs
                            chainDict['nowmTasklist'][-1]['ConfigCacheID']='%s/%s.py'%(dir,step)
                            chainDict['nowmTasklist'][-1]['GlobalTag']=chainDict['nowmTasklist'][-1]['nowmIO']['GT'] # copy to the proper parameter name
                            chainDict['GlobalTag']=chainDict['nowmTasklist'][-1]['nowmIO']['GT'] #set in general to the last one of the chain
                            if 'pileup' in chainDict['nowmTasklist'][-1]['nowmIO']:
                                chainDict['nowmTasklist'][-1]['MCPileup']=chainDict['nowmTasklist'][-1]['nowmIO']['pileup']
                            if '--pileup ' in s[2][index]:      # catch --pileup (scenarion) and not --pileup_ (dataset to be mixed) => works also making PRE-MIXed dataset
                                processStrPrefix='PU_'          # take care of pu overlay done with GEN-SIM mixing
                                if (  s[2][index].split()[  s[2][index].split().index('--pileup')+1 ]  ).find('25ns')  > 0 :
                                    processStrPrefix='PU25ns_'
                                elif   (  s[2][index].split()[  s[2][index].split().index('--pileup')+1 ]  ).find('50ns')  > 0 :
                                    processStrPrefix='PU50ns_'
                            if 'DIGIPREMIX_S2' in s[2][index] : # take care of pu overlay done with DIGI mixing of premixed events
                                if s[2][index].split()[ s[2][index].split().index('--pileup_input')+1  ].find('25ns')  > 0 :
                                    processStrPrefix='PUpmx25ns_'
                                elif s[2][index].split()[ s[2][index].split().index('--pileup_input')+1  ].find('50ns')  > 0 :
                                    processStrPrefix='PUpmx50ns_'

                            if acqEra:
                                #chainDict['AcquisitionEra'][step]=(chainDict['CMSSWVersion']+'-PU_'+chainDict['nowmTasklist'][-1]['GlobalTag']).replace('::All','')+thisLabel
                                chainDict['AcquisitionEra'][step]=chainDict['CMSSWVersion']
                                chainDict['ProcessingString'][step]=processStrPrefix+chainDict['nowmTasklist'][-1]['GlobalTag'].replace('::All','')+thisLabel
                            else:
                                #chainDict['nowmTasklist'][-1]['AcquisitionEra']=(chainDict['CMSSWVersion']+'-PU_'+chainDict['nowmTasklist'][-1]['GlobalTag']).replace('::All','')+thisLabel
                                chainDict['nowmTasklist'][-1]['AcquisitionEra']=chainDict['CMSSWVersion']
                                chainDict['nowmTasklist'][-1]['ProcessingString']=processStrPrefix+chainDict['nowmTasklist'][-1]['GlobalTag'].replace('::All','')+thisLabel

                            if (self.batchName):
                                chainDict['nowmTasklist'][-1]['Campaign'] = chainDict['nowmTasklist'][-1]['AcquisitionEra']+self.batchName

                            # specify different ProcessingString for double miniAOD dataset
                            if ('DBLMINIAODMCUP15NODQM' in step): 
                                chainDict['nowmTasklist'][-1]['ProcessingString']=chainDict['nowmTasklist'][-1]['ProcessingString']+'_miniAOD' 

                            if( chainDict['nowmTasklist'][-1]['Multicore'] ):
                                # the scaling factor of 1.2GB / thread is empirical and measured on a SECOND round of tests with PU samples
                                # the number of threads is NO LONGER assumed to be the same for all tasks
                                # https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/3509/1/1/1.html
                                # now change to 1.5GB / additional thread according to discussion:
                                # https://hypernews.cern.ch/HyperNews/CMS/get/relval/4817/1/1.html
#                                chainDict['nowmTasklist'][-1]['Memory'] = 3000 + int( chainDict['nowmTasklist'][-1]['Multicore']  -1 )*1500
                                chainDict['nowmTasklist'][-1]['Memory'] = self.memoryOffset + int( chainDict['nowmTasklist'][-1]['Multicore']  -1 ) * self.memPerCore

                        index+=1
                    #end of loop through steps
                    chainDict['RequestString']='RV'+chainDict['CMSSWVersion']+s[1].split('+')[0]
                    if processStrPrefix or thisLabel:
                        chainDict['RequestString']+='_'+processStrPrefix+thisLabel

### PrepID
                    chainDict['PrepID'] = chainDict['CMSSWVersion']+'__'+self.batchTime+'-'+s[1].split('+')[0]
                    if(self.batchName):
                        chainDict['PrepID'] = chainDict['CMSSWVersion']+self.batchName+'-'+s[1].split('+')[0]
                        if( 'HIN' in self.batchName ):
                            chainDict['SubRequestType'] = "HIRelVal"
                        
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

            # agreed changes for wm injection:
            # - Campaign: *optional* string during creation. It will default to AcqEra value if possible.  
            #             Otherwise it will be empty.
            # - AcquisitionEra: *mandatory* string at request level during creation. *optional* string
            #                   at task level during creation. "optional" during assignment.
            # - ProcessingString: *mandatory* string at request level during creation. *optional* string
            #                     at task level during creation. "optional" during assignment.
            # - ProcessingVersion: *optional* during creation (default 1). *optional* during assignment.
            # 
            # Which requires following changes here:
            #  - reset Global AcuisitionEra, ProcessingString to be the one in the first task
            #  - and also Campaign to be always the same as the AcquisitionEra

            if acqEra:
                chainDict['AcquisitionEra'] = chainDict['AcquisitionEra'].values()[0] 
                chainDict['ProcessingString'] = chainDict['ProcessingString'].values()[0]
            else:
                chainDict['AcquisitionEra'] = chainDict['nowmTasklist'][0]['AcquisitionEra']
                chainDict['ProcessingString'] = chainDict['nowmTasklist'][0]['ProcessingString']
                
#####  batch name appended to Campaign name
#            chainDict['Campaign'] = chainDict['AcquisitionEra']
            chainDict['Campaign'] = chainDict['AcquisitionEra']+self.batchName
               
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
                                            d['ConfigCacheUrl']
                                            )
                    print d[it]['ConfigCacheID']," uploaded to couchDB for",str(n),"with ID",couchID
                    d[it]['ConfigCacheID']=couchID
                if it =='DQMConfigCacheID':
                    couchID=self.uploadConf(d['DQMConfigCacheID'],
                                            str(n)+'harvesting',
                                            d['ConfigCacheUrl']
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
                print "...........",n,"submitted"
                random_sleep()
            

        
