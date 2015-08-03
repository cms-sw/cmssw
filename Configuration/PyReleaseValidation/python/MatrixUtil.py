class Matrix(dict):
    def __setitem__(self,key,value):
        if key in self:
            print "ERROR in Matrix"
            print "overwritting",key,"not allowed"
        else:
            self.update({float(key):WF(float(key),value)})

    def addOverride(self,key,override):
        self[key].addOverride(override)
            
#the class to collect all possible steps
class Steps(dict):
    def __setitem__(self,key,value):
        if key in self:
            print "ERROR in Step"
            print "overwritting",key,"not allowed"
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
            print 'steps',s,stepsDict[s]
            steps.append(stepsDict[s])

    
InputInfoNDefault=2000000    
class InputInfo(object):
    def __init__(self,dataSet,label='',run=[],files=1000,events=InputInfoNDefault,split=10,location='CAF',ib_blacklist=None,ib_block=None) :
        self.run = run
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet
        self.split = split
        self.ib_blacklist = ib_blacklist
        self.ib_block = ib_block
        
    def das(self, das_options):
        if len(self.run) is not 0:
            command = ";".join(["das_client.py %s --query '%s'" % (das_options, query) for query in self.queries()])
            command = "({0})".format(command)
        else:
            command = "das_client.py %s --query '%s'" % (das_options, self.queries()[0])
       
        # Run filter on DAS output 
        if self.ib_blacklist:
            command += " | grep -E -v "
            command += " ".join(["-e '{0}'".format(pattern) for pattern in self.ib_blacklist])
        command += " | sort -u"
        return command

    def lumiRanges(self):
        if len(self.run) != 0:
            return "echo '{\n"+",".join(('"%d":[[1,268435455]]\n'%(x,) for x in self.run))+"}'"
        return None

    def queries(self):
        query_by = "block" if self.ib_block else "dataset"
        query_source = "{0}#{1}".format(self.dataSet, self.ib_block) if self.ib_block else self.dataSet
        if len(self.run) is not 0:
            return ["file {0}={1} run={2} site=T2_CH_CERN".format(query_by, query_source, query_run) for query_run in self.run]
        else:
            return ["file {0}={1} site=T2_CH_CERN".format(query_by, query_source)]

    def __str__(self):
        if self.ib_block:
            return "input from: {0} with run {1}#{2}".format(self.dataSet, self.ib_block, self.run)
        return "input from: {0} with run {1}".format(self.dataSet, self.run)
    
# merge dictionaries, with prioty on the [0] index
def merge(dictlist,TELL=False):
    import copy
    last=len(dictlist)-1
    if TELL: print last,dictlist
    if last==0:
        # ONLY ONE ITEM LEFT
        return copy.copy(dictlist[0])
    else:
        reducedlist=dictlist[0:max(0,last-1)]
        if TELL: print reducedlist
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
    if TELL: print "original dict, BEF: %s"%d
    del e[key]
    if TELL: print "copy-removed dict, AFT: %s"%e
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


