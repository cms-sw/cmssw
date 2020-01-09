from __future__ import print_function
class Matrix(dict):
    def __setitem__(self,key,value):
        if key in self:
            print("ERROR in Matrix")
            print("overwriting",key,"not allowed")
        else:
            self.update({float(key):WF(float(key),value)})

    def addOverride(self,key,override):
        self[key].addOverride(override)
            
#the class to collect all possible steps
class Steps(dict):
    def __setitem__(self,key,value):
        if key in self:
            print("ERROR in Step")
            print("overwriting",key,"not allowed")
            import sys
            sys.exit(-9)
        else:
            self.update({key:value})
            # make the python file named <step>.py
            #if not '--python' in value:                self[key].update({'--python':'%s.py'%(key,)})

    def overwrite(self,keypair):
        value=self[keypair[1]]
        self.update({keypair[0]:value})
        
class WF(list):
    def __init__(self,n,l):
        self.extend(l)
        self.num=n
        #the actual steps of this WF
        self.steps=[]
        self.overrides={}
    def addOverride(self,overrides):
        self.overrides=overrides
        
    def interpret(self,stepsDict):
        for s in self:
            print('steps',s,stepsDict[s])
            steps.append(stepsDict[s])
    


def expandLsInterval(lumis):
    return range(lumis[0],(lumis[1]+1))

from DPGAnalysis.Skims.golden_json_2015 import * 
jsonFile2015 = findFileInPath("DPGAnalysis/Skims/data/Cert_13TeV_16Dec2015ReReco_Collisions15_25ns_50ns_JSON.txt")
jsonFile2016 = findFileInPath("DPGAnalysis/Skims/data/Cert_271036-274240_13TeV_PromptReco_Collisions16_JSON.txt")

import json
with open(jsonFile2015) as data_file:
    data_json2015 = json.load(data_file)

with open(jsonFile2016) as data_file:
    data_json2016 = json.load(data_file)

# return a portion of the 2015 golden json
# LS for a full run by default; otherwise a subset of which you determined the size
def selectedLS(list_runs=[],maxNum=-1,l_json=data_json2015):
    # print "maxNum is %s"%(maxNum)
    if not isinstance(list_runs[0], int):
        print("ERROR: list_runs must be a list of integers")
        return None
    local_dict = {}
    ls_count = 0

    for run in list_runs:
        if str(run) in l_json.keys():
            # print "run %s is there"%(run)
            runNumber = run
            # print "Doing lumi-section selection for run %s: "%(run)
            for LSsegment in l_json[str(run)] :
                # print LSsegment
                ls_count += (LSsegment[-1] - LSsegment[0] + 1)
                if (ls_count > maxNum) & (maxNum != -1):
                    break
                    # return local_dict
                if runNumber in local_dict.keys():
                    local_dict[runNumber].append(LSsegment)
                else: 
                    local_dict[runNumber] = [LSsegment]
                # print "total LS so far  %s    -   grow %s"%(ls_count,local_dict)
            #local_dict[runNumber] = [1,2,3]
        else:
            print("run %s is NOT present in json %s\n\n"%(run, l_json))
        # print "++    %s"%(local_dict)

    if ( len(local_dict) > 0 ) :
        return local_dict
    else :
        print("No luminosity section interval passed the json and your selection; returning None")
        return None

# print "\n\n\n THIS IS WHAT I RETURN: %s \n\n"%( selectedLS([251244,251251]) )




InputInfoNDefault=2000000    
class InputInfo(object):
    def __init__(self,dataSet,dataSetParent='',label='',run=[],ls={},files=1000,events=InputInfoNDefault,split=10,location='CAF',ib_blacklist=None,ib_block=None) :
        self.run = run
        self.ls = ls
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet
        self.split = split
        self.ib_blacklist = ib_blacklist
        self.ib_block = ib_block
        self.dataSetParent = dataSetParent
        
    def das(self, das_options, dataset):
        if len(self.run) is not 0 or self.ls:
            queries = self.queries(dataset)[:3]
            if len(self.run) != 0:
              command = ";".join(["dasgoclient %s --query '%s'" % (das_options, query) for query in queries])
            else:
              lumis = self.lumis()
              commands = []
              while queries:
                commands.append("dasgoclient %s --query 'lumi,%s' --format json | das-selected-lumis.py %s " % (das_options, queries.pop(), lumis.pop()))
              command = ";".join(commands)
            command = "({0})".format(command)
        else:
            command = "dasgoclient %s --query '%s'" % (das_options, self.queries(dataset)[0])
       
        # Run filter on DAS output 
        if self.ib_blacklist:
            command += " | grep -E -v "
            command += " ".join(["-e '{0}'".format(pattern) for pattern in self.ib_blacklist])
        from os import getenv
        if getenv("CMSSW_USE_IBEOS","false")=="true": return command + " | ibeos-lfn-sort"
        return command + " | sort -u"

    def lumiRanges(self):
        if len(self.run) != 0:
            return "echo '{\n"+",".join(('"%d":[[1,268435455]]\n'%(x,) for x in self.run))+"}'"
        if self.ls :
            return "echo '{\n"+",".join(('"%d" : %s\n'%( int(x),self.ls[x]) for x in self.ls.keys()))+"}'"
        return None

    def lumis(self):
      query_lumis = []
      if self.ls:
        for run in self.ls.keys():
          run_lumis = []
          for rng in self.ls[run]: run_lumis.append(str(rng[0])+","+str(rng[1]))
          query_lumis.append(":".join(run_lumis))
      return query_lumis

    def queries(self, dataset):
        query_by = "block" if self.ib_block else "dataset"
        query_source = "{0}#{1}".format(dataset, self.ib_block) if self.ib_block else dataset

        if self.ls :
            the_queries = []
            #for query_run in self.ls.keys():
            # print "run is %s"%(query_run)
            # if you have a LS list specified, still query das for the full run (multiple ls queries take forever)
            # and use step1_lumiRanges.log to run only on LS which respect your selection

            # DO WE WANT T2_CERN ?
            return ["file {0}={1} run={2}".format(query_by, query_source, query_run) for query_run in self.ls.keys()]
            #return ["file {0}={1} run={2} site=T2_CH_CERN".format(query_by, query_source, query_run) for query_run in self.ls.keys()]


                # 
                #for a_range in self.ls[query_run]:
                #    # print "a_range is %s"%(a_range)
                #    the_queries +=  ["file {0}={1} run={2} lumi={3} ".format(query_by, query_source, query_run, query_ls) for query_ls in expandLsInterval(a_range) ]
            #print the_queries
            return the_queries

        if len(self.run) is not 0:
            return ["file {0}={1} run={2} site=T2_CH_CERN".format(query_by, query_source, query_run) for query_run in self.run]
            #return ["file {0}={1} run={2} ".format(query_by, query_source, query_run) for query_run in self.run]
        else:
            return ["file {0}={1} site=T2_CH_CERN".format(query_by, query_source)]
            #return ["file {0}={1} ".format(query_by, query_source)]

    def __str__(self):
        if self.ib_block:
            return "input from: {0} with run {1}#{2}".format(self.dataSet, self.ib_block, self.run)
        return "input from: {0} with run {1}".format(self.dataSet, self.run)

    
# merge dictionaries, with prioty on the [0] index
def merge(dictlist,TELL=False):
    import copy
    last=len(dictlist)-1
    if TELL: print(last,dictlist)
    if last==0:
        # ONLY ONE ITEM LEFT
        return copy.copy(dictlist[0])
    else:
        reducedlist=dictlist[0:max(0,last-1)]
        if TELL: print(reducedlist)
        # make a copy of the last item
        d=copy.copy(dictlist[last])
        # update with the last but one item
        d.update(dictlist[last-1])
        # and recursively do the rest
        reducedlist.append(d)
        return merge(reducedlist,TELL)

def remove(d,key,TELL=False):
    import copy
    e = copy.deepcopy(d)
    if TELL: print("original dict, BEF: %s"%d)
    del e[key]
    if TELL: print("copy-removed dict, AFT: %s"%e)
    return e


#### Standard release validation samples ####

stCond={'--conditions':'auto:run1_mc'}
def Kby(N,s):
    return {'--relval':'%s000,%s'%(N,s)}
def Mby(N,s):
    return {'--relval':'%s000000,%s'%(N,s)}

def changeRefRelease(steps,listOfPairs):
    for s in steps:
        if ('INPUT' in steps[s]):
            oldD=steps[s]['INPUT'].dataSet
            for (ref,newRef) in listOfPairs:
                if  ref in oldD:
                    steps[s]['INPUT'].dataSet=oldD.replace(ref,newRef)
        if '--pileup_input' in steps[s]:
            for (ref,newRef) in listOfPairs:
                if ref in steps[s]['--pileup_input']:
                    steps[s]['--pileup_input']=steps[s]['--pileup_input'].replace(ref,newRef)
        
def addForAll(steps,d):
    for s in steps:
        steps[s].update(d)


def genvalid(fragment,d,suffix='all',fi='',dataSet=''):
    import copy
    c=copy.copy(d)
    if suffix:
        c['-s']=c['-s'].replace('genvalid','genvalid_'+suffix)
    if fi:
        c['--filein']='lhe:%d'%(fi,)
    if dataSet:
        c['--filein']='das:%s'%(dataSet,)
    c['cfg']=fragment
    return c


