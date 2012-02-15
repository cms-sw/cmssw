import sys
import json
import os
import copy


class MatrixInjector(object):

    def __init__(self,mode='init'):
        self.count=1040
        self.testMode=(mode!='submit')
        self.defaultChain={
            "AcquisitionEra": "ReleaseValidation",            #Acq Era
            "Requestor": "vlimant@cern.ch",                   #Person responsible
            "CMSSWVersion": os.getenv('CMSSW_VERSION'),       #CMSSW Version (used for all tasks in chain)
            "ScramArch": os.getenv('SCRAM_ARCH'),             #Scram Arch (used for all tasks in chain)
            "ProcessingVersion": "v1",                        #Processing Version (used for all tasks in chain)
            "GlobalTag": None,                                #Global Tag (used for all tasks)
            "CouchURL": "http://couchserver.cern.ch",         #URL of CouchDB containing Config Cache
            "CouchDBName": "config_cache",                    #Name of Couch Database containing config cache
            #- Will contain all configs for all Tasks
            "SiteWhitelist" : ["T1_CH_CERN", "T1_US_FNAL"],   #Site whitelist
            "TaskChain" : None,                                  #Define number of tasks in chain.
            "nowmTasklist" : []  #a list of tasks as we put them in
            }

        self.defaultScratch={
            "TaskName" : None,                            #Task Name
            "ConfigCacheID" : None,                   #Generator Config id
            "SplittingAlgorithm"  : "EventBased",             #Splitting Algorithm
            "SplittingArguments" : {"events_per_job" : 250},  #Size of jobs in terms of splitting algorithm
            "RequestSizeEvents" : 10000,                      #Total number of events to generate
            "Seeding" : "Automatic",                          #Random seeding method
            "PrimaryDataset" : None,                          #Primary Dataset to be created
            }
        self.defaultInput={
            "TaskName" : "DigiHLT",                                      #Task Name
            "ConfigCacheID" : None,                                      #Processing Config id
            "InputDataset" : None,                                       #Input Dataset to be processed
            "SplittingAlgorithm"  : "FileBased",                        #Splitting Algorithm
            "SplittingArguments" : {"files_per_job" : 1},               #Size of jobs in terms of splitting algorithm
            }
        self.defaultTask={
            "TaskName" : None,                                 #Task Name
            "InputTask" : None,                                #Input Task Name (Task Name field of a previous Task entry)
            "InputFromOutputModule" : None,                    #OutputModule name in the input task that will provide files to process
            "ConfigCacheID" : None,                            #Processing Config id
            "SplittingAlgorithm" : "FileBased",                #Splitting Algorithm
            "SplittingArguments" : {"files_per_job" : 1 },     #Size of jobs in terms of splitting algorithm
            "nowmIO": {}
            }

        self.chainDicts={}

    def prepare(self,mReader, directories, mode='init'):
        
        for (n,dir) in directories.items():
            chainDict=copy.deepcopy(self.defaultChain)
            print "inspecting",dir
            nextHasDSInput=None
            for (x,s) in mReader.workFlowSteps.items():
                #x has the format (num, prefix)
                #s has the format (num, name, commands, stepList)
                if x[0]==n:
                    #print "found",n,s[3]
                    for (index,step) in enumerate(s[3]):
                        if 'INPUT' in step or (not isinstance(s[2][index],str)):
                            nextHasDSInput=s[2][index]
                        else:
                            if (index==0):
                                #first step and not input -> gen part
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultScratch))
                                chainDict['nowmTasklist'][-1]['PrimaryDataset']='RelVal'+step
                                if not '--relval' in s[2][index]:
                                    print 'Impossible to create task from scratch'
                                    return -12
                                else:
                                    arg=s[2][index].split()
                                    ns=arg[arg.index('--relval')+1].split(',')
                                    chainDict['nowmTasklist'][-1]['RequestSizeEvents'] = ns[0]
                                    chainDict['nowmTasklist'][-1]['SplittingArguments']['events_per_job'] = ns[1]
                            elif nextHasDSInput:
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultInput))
                                chainDict['nowmTasklist'][-1]['InputDataset']=nextHasDSInput.dataSet
                                # get the run numbers or #events
                                if len(nextHasDSInput.run):
                                    chainDict['nowmTasklist'][-1]['RunWhitelist']=nextHasDSInput.run
                                nextHasDSInput=None
                            else:
                                #not first step and no inputDS
                                chainDict['nowmTasklist'].append(copy.deepcopy(self.defaultTask))                                
                            #print step
                            chainDict['nowmTasklist'][-1]['TaskName']=step
                            try:
                                chainDict['nowmTasklist'][-1]['nowmIO']=json.loads(open('%s/%s.io'%(dir,step)).read())
                            except:
                                print "Failed to find",'%s/%s.io'%(dir,step),".The workflows were probably not run on cfg not created"
                                return -15
                            chainDict['nowmTasklist'][-1]['ConfigCacheID']='%s/%s.py'%(dir,step)
                            chainDict['GlobalTag']=chainDict['nowmTasklist'][-1]['nowmIO']['GT']
                            
            #wrap up for this one
            #print 'wrapping up'
            chainDict['TaskChain']=len(chainDict['nowmTasklist'])
            #loop on the task list
            for i_second in reversed(range(len(chainDict['nowmTasklist']))):
            #for t_second in reversed(chainDict['nowmTasklist']):
                t_second=chainDict['nowmTasklist'][i_second]
                #print "t_second taskname", t_second['TaskName']
                if 'primary' in t_second['nowmIO']:
                    #print t_second['nowmIO']['primary']
                    primary=t_second['nowmIO']['primary'][0].replace('file:','')
                    #for t_input in reversed(chainDict['nowmTasklist']):
                    for i_input in reversed(range(0,i_second)):
                        t_input=chainDict['nowmTasklist'][i_input]
                        for (om,o) in t_input['nowmIO'].items():
                            if primary in o:
                                #print "found",primary,"procuced by",om,"of",t_input['TaskName']
                                t_second['InputTask'] = t_input['TaskName']
                                t_second['InputFromOutputModule'] = om
                                #print 't_second',t_second
                                break
            for (i,t) in enumerate(chainDict['nowmTasklist']):
                t.pop('nowmIO')
                chainDict['Task%d'%(i+1)]=t

                                
            chainDict.pop('nowmTasklist')
            self.chainDicts[n]=chainDict
        return 0

    def uploadConf(self,filePath):
        if self.testMode:
            self.count+=1
            return self.count
        else:
            return 0
    
    def upload(self):
        for (n,d) in self.chainDicts.items():
            #look for toload:
            #upload it get couchID
            #replace the couchID
            for it in d:
                if it.startswith("Task") and it!='TaskChain':
                    couchID=self.uploadConf(d[it]['ConfigCacheID'])
                    print "uploading",d[it]['ConfigCacheID'],"to couchDB for",str(n),"got ID",couchID
                    d[it]['ConfigCacheID']=couchID
            
    def submit(self):
        import pprint
        for (n,d) in self.chainDicts.items():
            if self.testMode:
                print "Only viewing request",n
                print pprint.pprint(d)
            else:
                #submit to wmagent each dict
                print "submitting",n
                #do submit

            

        
