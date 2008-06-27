#import cmsconfigure as cms
import FWCore.ParameterSet.parsecf.pyparsing as pp
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.DictTypes import SortedKeysDict
from Mixins import PrintOptions
import copy
# Questions
#  If an include includes a parameter already defined is that an error?
#  If a 'using' block includes a parameter already defined, is that an error?
#    No, and the 'using' wins. But what is the intended behavior?
#  If two includes include the same parameter, is that an error?
#  if two 'using' blocks include the same parameter, is that an error?
#  How does 'using' with a block differ from 'using' with a PSet?
#    there is no difference except the PSet will wind up in the parameter set database
#  if we have two 'block' statements with the same name in same scope is that an error?
#    Answer: no and the second block wins
#    NOTE: two of 'any' type with the same label is presently allowed!
#    UPDATE: labels are not enforced to be unique in the C++ parser
#    UPDATE of UPDATE: this has been fixed

#Processing order
#  1) check for multiple inclusion of the same block can be done once the block is 'closed'
#     the check must be done recursively
#     NOTE: circular inclusions of a file are an error
#  2)

#keeps track of which files we are parsing
_fileStack = ["{config}"]

def _validateLabelledList(params):
    """enforces the rule that no label can be used more than once
    and if an include is done more than once we remove the duplicates
    """
    l = params[:]
    l.sort( lambda x,y:cmp(x[0],y[0]))
    previous=None
    toRemove = []
    for item in l:
        if previous and item[0]==previous:
            if type(item[1]) == _IncludeNode:
                toRemove.append(item[0])
            elif hasattr(item[1],'multiplesAllowed') and item[1].multiplesAllowed:
                continue
            else:
                raise RuntimeError("multiple items found with label:"+item[0])
        previous = item[0]
    for remove in toRemove:
        for index,item in enumerate(params):
            if item[0] == remove:
                del params[index]
                break
    return params


class _DictAdapter(object):
    def __init__(self,d, addSource=False):
        """Lets a dictionary be looked up by attributes"""
        #copy 'd' since we need to be able to lookup a 'source' by
        # it's type to do replace but we do NOT want to add it by its
        # type to the final Process
        self.__dict__['d'] = d
        if addSource and self.d.has_key('source'):
            self.d[d['source'].type_()]=d['source']
    def __setattr__(self,name,value):
        self.d[name]=value
    def __getattr__(self,name):
        #print 'asked for '+name
        return self.d[name]

class _DictCopyAdapter(object):
    def __init__(self,d):
        #copy 'd' since we need to be able to lookup a 'source' by
        # it's type to do replace but we do NOT want to add it by its
        # type to the final Process
        self.d = d.copy()
        if self.d.has_key('source'):
            self.d[d['source'].type_()]=d['source']
    def __getattr__(self,name):
        #print 'asked for '+name
        return self.d[name]


#setup a factory to open the file, we can redirect this for testing
class _IncludeFile(file):
    def __init__(self,filename):
        """Searches for the 'filename' using the appropriate paths"""
        import os.path
        from os import environ
        try:
            paths = environ['CMSSW_SEARCH_PATH']
        except KeyError:
            raise RuntimeError("The environment variable 'CMSSW_SEARCH_PATH' must be set for include to work")
        lpaths = paths.split(':')
        lpaths.append('.')
        f = None
        for path in lpaths:
            path +='/'+filename
            if os.path.exists(path):
                f=path
                break
        if f is None:
            raise RuntimeError("Unable to find file '"+filename+"' using the search path ${'CMSSW_SEARCH_PATH'} \n"
                               +paths)
        super(_IncludeFile,self).__init__(path,'r')
_fileFactory = _IncludeFile

def _findAndHandleParameterIncludes(values):
        return _findAndHandleParameterIncludesRecursive(values,set(),set())
def _findAndHandleParameterIncludesRecursive(values,otherFiles,recurseFiles):
    newValues = []
    for l,v in values:
        if isinstance(v,_IncludeNode):
            #newValues.extend(_handleParameterInclude(v.filename,otherFiles))
            newValues.extend(v.extract(l, otherFiles,
                                       recurseFiles,
                                       onlyParameters.parseFile,
                                       _validateLabelledList,
                                       _findAndHandleParameterIncludesRecursive))
        else:
            newValues.append((l,v))
    return newValues

def _findAndHandleProcessBlockIncludes(values):
        return _findAndHandleProcessBlockIncludesRecursive(values,set(),set())
def _findAndHandleProcessBlockIncludesRecursive(values,otherFiles,recurseFiles):
    newValues = []
    for l,v in values:
        if isinstance(v,_IncludeNode):
            #newValues.extend(_handleParameterInclude(v.filename,otherFiles))
            newValues.extend(v.extract(l, otherFiles,
                                       recurseFiles,
                                       onlyProcessBody.parseFile,
                                       _validateLabelledList,
                                       _findAndHandleProcessBlockIncludesRecursive))
        else:
            newValues.append((l,v))
    return newValues

def _handleInclude(fileName,otherFiles,recurseFiles,parser,validator,recursor):
    """reads in the file with name 'fileName' making sure it does not recursively include itself
    by looking in 'otherFiles' then applies the 'parser' to the contents of the file,
    runs the validator and then applies the recursor to see if other files must now be included"""
    global _fileStack
    if fileName in recurseFiles:
        raise RuntimeError('the file '+fileName+' eventually includes itself')
    if fileName in otherFiles:
        return list()
    newRecurseFiles = recurseFiles.copy()
    newRecurseFiles.add(fileName)
    otherFiles.add(fileName)
    _fileStack.append(fileName)
    try:
        factory = _fileFactory
        f = factory(fileName)
        try:
            values = parser(f)
            values = validator(values)
            values =recursor(values,otherFiles,newRecurseFiles)
        except pp.ParseException, e:
            raise RuntimeError('include file '+fileName+' had the parsing error \n'+str(e))
        except Exception, e:
            raise RuntimeError('include file '+fileName+' had the error \n'+str(e))
        try:
            values = validator(values)
        except Exception, e:
            raise RuntimeError('after including all other files, include file '+fileName+' had the error \n'+str(e))
        return values
    finally:
        _fileStack.pop()
        
def _handleUsing(using,otherUsings,process,allUsingLabels):
    """recursively go through the using blocks and return all the contained valued"""
    if using.value() in otherUsings:
        raise pp.ParseFatalException(using.s,using.loc,
    "the using.labelled '"+using.value()+"' recursively uses itself"+
    "\n from file "+using.file)
    allUsingLabels.add(using.value())
    values = []
    valuesFromOtherUsings=[]
    otherUsings = otherUsings.copy()
    otherUsings.add(using.value())
    if using.value() not in process:
        raise pp.ParseFatalException(using.s,using.loc,
                "the using.labelled '"+using.value()+"' does not correspond to a known block or PSet"
                +"\n from file "+using.file)
    d = process[using.value()].__dict__
    usingLabels=[]
    for label,param in (x for x in d.iteritems() if isinstance(x[1],cms._ParameterTypeBase)):
        if isinstance(param,cms.UsingBlock):
            newValues=_handleUsing(param,otherUsings,process,allUsingLabels)
            valuesFromOtherUsings.extend( newValues)
            values.extend(newValues)
            usingLabels.append(label)
        else:
            values.append((label,copy.deepcopy(param)))
    for label in usingLabels:
        #remove the using nodes
        delattr(process[using.value()],label)
    for plabel,param in valuesFromOtherUsings:
        item = process[using.value()]
        if hasattr(item,plabel):
            raise pp.ParseFatalException(using.s,using.loc,
                "the using labelled '"+using.value()+"' tried to add the label '"+
                plabel+"' which already exists in this block"
                +"\n from file "+using.file)
        setattr(item,plabel,param)
    return values

def _findAndHandleUsingBlocksRecursive(label,item,process,allUsingLabels):
    otherUsings = set( (label,))
    values = []
    usingLabels = []
    usingForValues = []
    for tempLabel,param in item.__dict__.iteritems():
        if isinstance(param,cms.UsingBlock):
            oldSize = len(values)
            values.extend(_handleUsing(param,otherUsings,process,allUsingLabels))
            usingLabels.append(tempLabel)
            usingForValues.extend([param]*(len(values)-oldSize))
        elif isinstance(param,cms._Parameterizable):
            _findAndHandleUsingBlocksRecursive(tempLabel,param,process,allUsingLabels)
        elif isinstance(param,cms.VPSet):
            for pset in param:
                _findAndHandleUsingBlocksRecursive(tempLabel,pset,process,allUsingLabels)
    for tempLabel in usingLabels:
        delattr(item,tempLabel)
    for index,(plabel,param) in enumerate(values):
        if hasattr(item,plabel):
            using = usingForValues[index]
            raise pp.ParseFatalException(using.s,using.loc,
                "the using labelled '"+using.value()+"' tried to add the label '"+
                plabel+"' which already exists in this block"
                +"\n from file "+using.file)
        setattr(item,plabel,param)

def _findAndHandleProcessUsingBlock(values):
    d=dict(values)
    allUsingLabels = set()
    for label,item in d.iteritems():
        if isinstance(item,cms._Parameterizable):
            _findAndHandleUsingBlocksRecursive(label,item,d,allUsingLabels)
        elif isinstance(item,cms.VPSet):
            for pset in item:
                _findAndHandleUsingBlocksRecursive(label,pset,d,allUsingLabels)
    return allUsingLabels

def _flagUsingBlocks(values):
    """Only flags whether they're resolvable or not"""
    d=dict(values)
    for label,item in d.iteritems():
        if isinstance(item,cms._Parameterizable):
            _flagUsingBlocksRecursive(item,d)
        elif isinstance(item,cms.VPSet):
            for pset in item:
                _flagUsingBlocksRecursive(pset,d)

def _flagUsingBlocksRecursive(item, d):
    for label,param in item.__dict__.iteritems():
        if isinstance(param,cms.UsingBlock):
            if item.label in d.keys():
                item.isResolved = True
        elif isinstance(param,cms._Parameterizable):
            _flagUsingBlocksRecursive(label,param,process)
        elif isinstance(param,cms.VPSet):
            for pset in param:
                _flagUsingBlocksRecursive(label,pset,process)


def _badLabel(s,loc,expr,err):
    """a mal formed label was detected"""
    raise pp.ParseFatalException(s,loc,"inappropriate label name")

def _makeParameter(s,loc,toks):
    """create the appropriate parameter object from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    if not hasattr(cms,toks[0][0]):
        raise pp.ParseFatalException(s,loc,'unknown parameter type '+toks[0][0])
    ptype = getattr(cms,toks[0][0])
    try:
        p = ptype._valueFromString(toks[0][2])
    except Exception,e:
        raise pp.ParseFatalException(s,loc,
                "failed to parse parameter '"+toks[0][1]+"' because of error\n"+str(e))
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledInputTag(s,loc,toks):
    """create an InputTag parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    if isinstance(toks[0][2], str):
        p = cms.InputTag(toks[0][2])
    else:
        values = list(iter(toks[0][2]))
        if len(values) == 1:
            values +=''
        p = cms.InputTag(*values)
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledVInputTag(s,loc,toks):
    """create an VInputTag parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]

    values = list(iter(toks[0][2]))
    items = []
    for x in values:
        if isinstance(x, str):
            items.append(cms.InputTag(x))
        else:
            items.append(cms.InputTag(*x))
    #items = [cms.InputTag(*x) for x in values]
    p = cms.VInputTag(*items)
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledEventID(s,loc,toks):
    """create an EventID parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    p = cms.EventID(int(toks[0][2][0]), int(toks[0][2][1]))
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledVEventID(s,loc,toks):
    """create an VEventID parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    values = list(iter(toks[0][2]))
    items = [cms.EventID(*x) for x in values]
    p = cms.VEventID(*items)
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledLuminosityBlockID(s,loc,toks):
    """create an EventID parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    p = cms.LuminosityBlockID(int(toks[0][2][0]), int(toks[0][2][1]))
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)


def _makeDictFromList(values):
    values = _validateLabelledList(values)
    values = _findAndHandleParameterIncludes(values)
    values = _validateLabelledList(values)
    return dict(values)

def _makePSetFromList(values):
    d = _makeDictFromList(values)
    p = cms.PSet(*[],**d)
    return p
    
def _makePSet(s,loc,toks):
    """create a PSet from the tokens"""
    values = list(iter(toks[0]))
    try:
        return _makePSetFromList(values)
    except Exception, e:
        raise pp.ParseFatalException(s,loc,"PSet contains the error \n"+str(e))
        

def _makeLabeledPSet(s,loc,toks):
    """create an PSet parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    p=_makePSet(s,loc,[toks[0][2]])
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

class _ObjectHolder(object):
    """If I return a VPSet directly to the parser it appears to 'eat it'"""
    def __init__(self,hold):
        self.hold = hold

def _makeVPSetFromList(values):
    items = [_makePSetFromList(x) for x in values]
    p = cms.VPSet(*items)
    return p    
def _makeVPSet(s,loc,toks):
    """create an VPSet from the tokens"""
    values = list(iter(toks[0]))
    try:
        p = _makeVPSetFromList(values)
        return _ObjectHolder(p)
    except Exception, e:
        raise pp.ParseFatalException(s,loc,"VPSet contains the error \n"+str(e))

def _makeLabeledVPSet(s,loc,toks):
    """create an VPSet parameter from the tokens"""
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    p = _makeVPSet(s,loc,(toks[0][2],)).hold
    if not tracked:
        cms.untracked(p)
    return (toks[0][1],p)

def _makeLabeledSecSource(s,loc,toks):
    tracked = True
    if len(toks[0])==4:
        tracked = False
        del toks[0][0]
    ss=toks[0][2]
    if not tracked:
        cms.untracked(ss)
    return (toks[0][1],ss)


#This is ugly but I need to know the labels used by all using statements
# in order to efficiently process the 
_allUsingLabels = set()
def _makeUsing(s,loc,toks):
    global _allUsingLabels
    _allUsingLabels.add(toks[0][1])
    #global _fileStack
    #file = _fileStack[-1]
    #TEMP:usings are hard, lets wait
    #raise pp.ParseFatalException(s,loc,"using not yet implemented")
    options = PrintOptions()
    return ('using_'+toks[0][1],cms.UsingBlock(toks[0][1], s, loc, _fileStack[-1]))

class _IncludeNode(cms._ParameterTypeBase):
    """For injection purposes, pretend this is a new parameter type
       then have a post process step which strips these out
    """
    def __init__(self,filename):
        self.filename = filename
    def parse(self, parser, validator):
        """Only the top-level nodes.  No recursion"""
        global _fileStack
        _fileStack.append(self.filename)
        try:
            #factory = _fileFactory
            #f = factory(fileName)
            f = _fileFactory(self.filename)
            try:
                values = parser(f)
                values = validator(values)
            except pp.ParseException, e:
                raise RuntimeError('include file '+self.filename+' had the parsing error \n'+str(e))
            except Exception, e:
                raise RuntimeError('include file '+self.filename+' had the error \n'+str(e))
            return values
        finally:
            _fileStack.pop()



    def extract(self, label, otherFiles,recurseFiles,parser,validator,recursor):
        """reads in the file with name 'fileName' making sure it does not recursively include itself
        by looking in 'otherFiles' then applies the 'parser' to the contents of the file,
        runs the validator and then applies the recursor to see if other files must now be included"""
        fileName = self.filename
        if fileName in recurseFiles:
            raise RuntimeError('the file '+fileName+' eventually includes itself')
        if fileName in otherFiles:
            return list()
        newRecurseFiles = recurseFiles.copy()
        newRecurseFiles.add(fileName)
        otherFiles.add(fileName)
        values = self.parse(parser, validator)
        values =recursor(values,otherFiles,newRecurseFiles)
        try:
            values = validator(values)
        except pp.ParseException, e:
           raise RuntimeError('after including all other files,include file '+fileName+' had the parsing error \n'+str(e))
        except Exception, e:
            raise RuntimeError('after including all other files, include file '+fileName+' had the error \n'+str(e))
        return values

    def pythonFileRoot(self):
        # translate, e.g., "SimMuon/DT/data/my-mod.cfi" to "SimMuon/DT/data/my_mod_cfi"
        return self.filename.replace('.','_').replace('-','_')
    def pythonFileName(self):
        return self.pythonFileRoot().replace('/data/','/python/').replace('/test/','/python/test/')+".py"
    def pythonModuleName(self):
        # we want something like "SimMuon.DT.mod_cfi"
        return self.pythonFileRoot().replace('/','.').replace('.data.','.')
    def dumpPython(self, options):
        if options.isCfg: 
            #return "import "+self.pythonModuleName()+"\nprocess.extend("+self.pythonModuleName()+")\n"
            return "process.load(\"" + self.pythonModuleName() + "\")\n"
        else:
            return "from "+self.pythonModuleName()+" import *"
    def createFile(self, overwrite):
        import os
        import os.path
        pythonName = self.pythonFileName()
        cmsswSrc = os.path.expandvars("$CMSSW_BASE/src/")
        cmsswReleaseSrc = os.path.expandvars("$CMSSW_RELEASE_BASE/src/")
        if overwrite or (not os.path.exists(cmsswSrc+pythonName) and \
                         not os.path.exists(cmsswReleaseSrc+pythonName)):
            # need to check out my own version
            cwd = os.getcwd()
            os.chdir(cmsswSrc)
            pythonDir = os.path.dirname(pythonName)
            #os.system("cvs co "+pythonDir)
            if overwrite and os.path.exists(pythonName):
                os.remove(pythonName)
            if not os.path.exists(pythonName):
                # have to make it myself
                if not os.path.exists(pythonDir):
                    #print "Making " + pythonDir
                    os.makedirs(pythonDir)
                    #os.system("scramv1 build")
                f=open(pythonName, 'w')
                f.write(dumpCff(self.filename))
                f.close()
                os.chdir(pythonDir)
            os.chdir(cwd)
          

class _IncludeFromNode(_IncludeNode):
    """An IncludeNode with a label, so it will only
       extract the named node, plus any IncludeNodes,
       which are presumed to be blocks"""
    def __init__(self,fromLabel, filename):
        super(_IncludeFromNode,self).__init__(filename)
        self._fromLabel = fromLabel
    def extract(self, newLabel, otherFiles,recurseFiles,parser,validator,recursor):
        import copy
        # First, expand everything, so blocks 
        # don't worry about re-parsing
        wasHere = (self.filename in otherFiles)
        if wasHere:
            otherFiles.remove(self.filename)
        expandedValues = _IncludeNode.extract(self, newLabel, otherFiles,recurseFiles,parser,validator,recursor)
        # one possible fix to the issue of whether the original gets included
        if not wasHere:
            otherFiles.remove(self.filename)
        found = False
        for l,v in expandedValues:
           if l == self._fromLabel:
               found = True
               # I don't know how to replace it, so I'll just have to copy
               # second possbile fix is to comment this out
               expandedValues.remove((l,v))
               expandedValues.append((newLabel, copy.deepcopy(v)))
        if not found:
            raise RuntimeError("the file "+self.filename+" does not contain a "+self._fromLabel
                                         +"\n from file "+_fileStack[-1])
        return expandedValues
    def dumpPythonAs(self, newLabel, options):
        result = "import "+self.pythonModuleName() +"\n"
        result += newLabel + " = "
        result += self.pythonModuleName() + '.' + self._fromLabel+".clone()\n"
        return result


def _makeInclude(s,loc,toks):
    return (toks[0][0],_IncludeNode(toks[0][0]))

letterstart =	pp.Word(pp.alphas,pp.srange("[a-zA-Z0-9\-_]"))
#dotdelimited    ([a-zA-Z]+[a-zA-Z0-9\-_]*+\.)+[a-zA-Z]+[a-zA-Z0-9\-_]*
#bangstart	\![a-zA-Z]+[a-zA-Z0-9\-_]*

#==================================================================
# Parameters
#==================================================================

#simple parameters are ones whose values do not have any delimiters
parameterValue = pp.Word(pp.alphanums+'.'+':'+'-'+'+')
simpleParameterType = pp.Keyword("bool")|pp.Keyword("int32")|pp.Keyword("uint32")|pp.Keyword("int64")|pp.Keyword("uint64")|pp.Keyword("double")
vSimpleParameterType = pp.Keyword("vint32")^pp.Keyword("vuint32")^pp.Keyword("vint64")^pp.Keyword("vuint64")^pp.Keyword("vdouble")
any = parameterValue | letterstart

_scopeBegin = pp.Suppress('{')
_scopeEnd = pp.Suppress('}')
label = letterstart.copy()
label.setFailAction(_badLabel)
untracked = pp.Optional('untracked')
#use the setFailAction to catch cases where internal to a label is an unsupported character ' bad$la = ' 
_equalTo = pp.Suppress('=').setFailAction(_badLabel)

simpleParameter = pp.Group(untracked+simpleParameterType+label
                           +_equalTo+any).setParseAction(_makeParameter)
vsimpleParameter = pp.Group(untracked+vSimpleParameterType+label+_equalTo
                            +_scopeBegin
                              +pp.Group(pp.Optional(pp.delimitedList(any)))
                            +_scopeEnd
                            ).setParseAction(_makeParameter)

def _handleString(s,loc,toks):
    #let python itself handle the string to get the substitutions right
    return eval(toks[0])
quotedString = pp.quotedString.copy().setParseAction(_handleString)
#quotedString = pp.quotedString.copy().setParseAction(pp.removeQuotes)
stringParameter = pp.Group(untracked+pp.Keyword('string')+label+_equalTo+
                           quotedString).setParseAction(_makeParameter)
vstringParameter =pp.Group(untracked+pp.Keyword("vstring")+label+_equalTo
                           +_scopeBegin
                             +pp.Group(pp.Optional(pp.delimitedList(quotedString)))
                           +_scopeEnd
                          ).setParseAction(_makeParameter)

fileInPathParameter = pp.Group(untracked+pp.Keyword('FileInPath')+label+_equalTo+
                           quotedString).setParseAction(_makeParameter)

inputTagFormat = pp.Group(letterstart+pp.Optional(pp.Suppress(':')+pp.Optional(pp.NotAny(pp.White())+pp.Word(pp.alphanums),"")+
                          pp.Optional(pp.Suppress(':')+pp.Optional(pp.NotAny(pp.White())+pp.Word(pp.alphanums)))))
anyInputTag = inputTagFormat|quotedString

#inputTagParameter = pp.Group(untracked+pp.Keyword('InputTag')+label+_equalTo+
#                             inputTagFormat
#                             ).setParseAction(_makeLabeledInputTag)
inputTagParameter = pp.Group(untracked+pp.Keyword('InputTag')+label+_equalTo+
                             anyInputTag
                             ).setParseAction(_makeLabeledInputTag)


vinputTagParameter =pp.Group(untracked+pp.Keyword("VInputTag")+label+_equalTo
                             +_scopeBegin
                               +pp.Group(pp.Optional(pp.delimitedList(anyInputTag)))
                             +_scopeEnd
                          ).setParseAction(_makeLabeledVInputTag)

eventIDParameter = pp.Group(untracked+pp.Keyword("EventID")+label+_equalTo 
                   + pp.Group( pp.Word(pp.nums) + pp.Suppress(':') + pp.Word(pp.nums) ) 
                   ).setParseAction(_makeLabeledEventID)

luminosityBlockIDParameter = pp.Group(untracked+pp.Keyword("LuminosityBlockID")+label+_equalTo
                   + pp.Group( pp.Word(pp.nums) + pp.Suppress(':') + pp.Word(pp.nums) )
                   ).setParseAction(_makeLabeledLuminosityBlockID)

#since PSet and VPSets can contain themselves, we must declare them as 'Forward'
PSetParameter = pp.Forward()
VPSetParameter = pp.Forward()
secsourceParameter = pp.Forward()
block = pp.Forward()
# need to tolerate internal blocks?
def _makeLabeledBlock(s,loc,toks):
    """create a PSet parameter from the tokens"""
    p=_makePSet(s,loc,[toks[0][2]])
    # They were tracked in the old system
    #p=cms.untracked(p)
    return (toks[0][1],p)

using = pp.Group(pp.Keyword("using")+letterstart).setParseAction(_makeUsing)
include = pp.Group(pp.Keyword("include").suppress()+quotedString).setParseAction(_makeInclude)

# blocks and includes need to be included in case they're in fragments
parameter = simpleParameter|stringParameter|vsimpleParameter|fileInPathParameter|vstringParameter|inputTagParameter|vinputTagParameter|eventIDParameter|luminosityBlockIDParameter|PSetParameter|VPSetParameter|secsourceParameter|block|include

scopedParameters = _scopeBegin+pp.Group(pp.ZeroOrMore(parameter|using|include))+_scopeEnd
#now we can actually say what PSet and VPSet are
block <<  pp.Group(untracked+pp.Keyword("block")+label+_equalTo+scopedParameters
                ).setParseAction(_makeLabeledBlock)

PSetParameter << pp.Group(untracked+pp.Keyword("PSet")+label+_equalTo+scopedParameters
                          ).setParseAction(_makeLabeledPSet)
VPSetParameter << pp.Group(untracked+pp.Keyword("VPSet")+label+_equalTo
                           +_scopeBegin
                               +pp.Group(pp.Optional(pp.delimitedList(scopedParameters)))
                           +_scopeEnd
                          ).setParseAction(_makeLabeledVPSet)

parameters = pp.OneOrMore(parameter)
parameters.ignore(pp.cppStyleComment)
parameters.ignore(pp.pythonStyleComment)


#==================================================================
# Plugins
#==================================================================

class _MakePlugin(object):
    def __init__(self,plugin):
        self.__plugin = plugin
    def __call__(self,s,loc,toks):
        global _fileStack
        type = toks[0][0]
        values = list(iter(toks[0][1]))
        try:
            values = _validateLabelledList(values)
            values = _findAndHandleParameterIncludes(values)
            values = _validateLabelledList(values)
        except Exception, e:
            raise pp.ParseFatalException(s,loc,type+" contains the error "+str(e)
                                         +"\n from file "+_fileStack[-1])
        d = dict(values)
        return self.__plugin(*[type],**d)
class _MakeFrom(object):
    def __init__(self,plugin):
        self.__plugin = plugin
    def __call__(self,s,loc,toks):
        global _fileStack
        label = toks[0][0]
        inc = toks[0][1]
        return _IncludeFromNode(label, inc[0])

def _replaceKeywordWithType(s,loc,toks):
    type = toks[0][1].type_()
    return (type,toks[0][1])

def _MakeESPrefer(s, loc, toks):
    value = None
    label = 'es_prefer_'+toks[0][1]
    if len(toks[0]) == 4:
        # type, label
        value = cms.ESPrefer(toks[0][2], toks[0][1]) 
    elif len(toks[0])==3:
        value = cms.ESPrefer(toks[0][1])
    else:
        print "Strange parse of es_prefer "+str(toks)
    return (label,value)

    
typeWithParameters = pp.Group(letterstart+scopedParameters)

#secsources are parameters but they behave like Plugins
secsourceParameter << pp.Group(untracked+pp.Keyword("secsource")+label+_equalTo
                               +typeWithParameters.copy().setParseAction(_MakePlugin(cms.SecSource))
                          ).setParseAction(_makeLabeledSecSource)

source = pp.Group(pp.Keyword("source")+_equalTo
                  +typeWithParameters.copy().setParseAction(_MakePlugin(cms.Source))
                 )
looper = pp.Group(pp.Keyword("looper")+_equalTo
                  +typeWithParameters.copy().setParseAction(_MakePlugin(cms.Looper))
                 )

service = pp.Group(pp.Keyword("service")+_equalTo
                   +typeWithParameters.copy().setParseAction(_MakePlugin(cms.Service))
                  ).setParseAction(_replaceKeywordWithType)

es_prefer = pp.Group(pp.Keyword("es_prefer")+pp.Optional(letterstart)+_equalTo+label+scopedParameters).setParseAction(_MakeESPrefer)


#for now, pretend all modules are filters since filters can function like
# EDProducer's or EDAnalyzers
module = pp.Group(pp.Suppress(pp.Keyword("module"))+label+_equalTo
                  +typeWithParameters.copy().setParseAction(_MakePlugin(cms.EDFilter))|
                  pp.Suppress(pp.Keyword("module"))+label+_equalTo
                  +pp.Group(label+pp.Group(pp.Keyword("from").suppress()+quotedString).setParseAction(_makeInclude)).setParseAction(_MakeFrom(cms.EDFilter)))

def _guessTypeFromClassName(regexp,type):
    return pp.Group(pp.Suppress(pp.Keyword('module'))+label+_equalTo
                             +pp.Group(pp.Regex(regexp)
                                       +scopedParameters
                                      ).setParseAction(_MakePlugin(type))
                             )
outputModuleGuess = _guessTypeFromClassName(r"[a-zA-Z]\w*OutputModule",cms.OutputModule)
producerGuess = _guessTypeFromClassName(r"[a-zA-Z]\w*Prod(?:ucer)?",cms.EDProducer)
analyzerGuess = _guessTypeFromClassName(r"[a-zA-Z]\w*Analyzer",cms.EDAnalyzer)

def _labelOptional(alabel,type,appendToLabel=''):
    def useTypeIfNoLabel(s,loc,toks):
        if len(toks[0])==2:
            alabel = toks[0][0]
            del toks[0][0]
        else:
            alabel = toks[0][0].type_()
        alabel +=appendToLabel
        return (alabel,toks[0][0])
    #NOTE: must use letterstart instead of label else get exception when no label
    return pp.Group(pp.Suppress(pp.Keyword(alabel))+pp.Optional(letterstart)+_equalTo
                              +typeWithParameters.copy().setParseAction(_MakePlugin(type))|
                              pp.Keyword(alabel).suppress()+pp.Optional(letterstart)+_equalTo+pp.Group(label+pp.Group(pp.Keyword("from").suppress()+quotedString).setParseAction(_makeInclude)).setParseAction(_MakeFrom(type))
                             ).setParseAction(useTypeIfNoLabel)

es_module = _labelOptional("es_module",cms.ESProducer)
es_source = _labelOptional("es_source",cms.ESSource)

plugin = source|looper|service|outputModuleGuess|producerGuess|analyzerGuess|module|es_module|es_source|es_prefer
plugin.ignore(pp.cppStyleComment)
plugin.ignore(pp.pythonStyleComment)

#==================================================================
# Paths
#==================================================================
#NOTE: I can't make the parser change a,b,c into (a,b),c only a,(b,c)
# so instead, I reverse the order of the tokens and then do the parsing
# and then build the parse tree from right to left
pathexp = pp.Forward()
# Really want either ! or -
_pathAtom = pp.Combine(pp.Optional("!")+pp.Optional("-")+letterstart)
#_pathAtom = pp.Combine(pp.Optional("!")+letterstart)
worker = (_pathAtom)^pp.Group(pp.Suppress(')')+pathexp+pp.Suppress('('))
pathseq = pp.Forward()
pathseq << pp.Group(worker + pp.ZeroOrMore(','+pathseq))
pathexp << pp.Group(pathseq + pp.ZeroOrMore('&'+pathexp))

class _LeafNode(object):
    def __init__(self,label):
        self.__isNot = False
        self.__isIgnore = False
        self._label = label
        if self._label[0]=='!':
            self._label=self._label[1:]
            self.__isNot = True
        elif self._label[0]=='-':
            self._label=self._label[1:]
            self.__isIgnore = True

    def __str__(self):
        v=''
        if self.__isNot:
            v='!'
        elif self.__isIgnore:
            v += '-'
        return v+self._label
    def make(self,process):
        #print getattr(process,self.__label).label()
        v = getattr(process,self._label)
        if self.__isNot:
            v= ~v
        elif self.__isIgnore:
            v= cms.ignore(v)
        return v
    def getLeaves(self, leaves):
        leaves.append(self)
    def dumpPython(self, options):
        result = ''
        if self.__isNot:
            result += '~'
        elif self.__isIgnore:
            result += 'cms.ignore('
        if options.isCfg:
            result += "process."
        result += self._label
        if self.__isIgnore:
            result += ')'
        return result

class _AidsOp(object):
    def __init__(self,left,right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+','+str(self.__right)+')'
    def getLeaves(self, leaves):
        self.__left.getLeaves(leaves)
        self.__right.getLeaves(leaves)
    def dumpPython(self, options):
        return self.__left.dumpPython(options)+'*'+self.__right.dumpPython(options)
    def make(self,process):
        left = self.__left.make(process)
        right = self.__right.make(process)
        return left*right

class _FollowsOp(object):
    def __init__(self,left,right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+'&'+str(self.__right)+')'
    def getLeaves(self, leaves):
        self.__left.getLeaves(leaves)
        self.__right.getLeaves(leaves)
    def dumpPython(self, options):
        return self.__left.dumpPython(options)+'+'+self.__right.dumpPython(options)
    def make(self,process):
        left = self.__left.make(process)
        right = self.__right.make(process)
        return left+right

def _buildTree(tree):
    #print 'tree = '+str(tree)
    if isinstance(tree,type('')):
        return _LeafNode(tree)
    if len(tree) == 1:
        return _buildTree(tree[0])
    assert(len(tree) == 3)
    left = _buildTree(tree[0])
    right = _buildTree(tree[2])
    theOp = _FollowsOp
    if ',' == tree[1]:
        theOp = _AidsOp
#    return [right,tree[1],left]
    return theOp(right,left)

def _parsePathInReverse(s,loc,toks):
    backwards = list(toks[0])
    backwards.reverse()
    return [_buildTree(pathexp.parseString(' '.join(backwards)))]

class _ModuleSeries(object):
    def __init__(self,topNode,s,loc,toks):
        global _fileStack
        #NOTE: nee to record what file we are from as well
        self.topNode = topNode
        self.forErrorMessage = (s,loc,toks,_fileStack[-1])
    def make(self,process):
        try:
            nodes = self.topNode.make(process)
            return self.factory()(nodes)
        except AttributeError, e:
            raise pp.ParseFatalException(self.forErrorMessage[0],
                                         self.forErrorMessage[1],
                                         self.type()+" '"
                                         +self.forErrorMessage[2][0][0]+
                                         "' contains the error: "
                                         +str(e)
                                         +"\n from file "+self.forErrorMessage[3])
        except Exception, e:
            raise pp.ParseFatalException(self.forErrorMessage[0],
                                         self.forErrorMessage[1],
                                         self.type()
                                         +" '"+self.forErrorMessage[2][0][0]
                                         +"' contains the error: "+str(e)
                                         +"\n from file "+self.forErrorMessage[3])
    def __str__(self):
        return str(self.topNode)
    def __repr__(self):
        options = PrintOptions()
        # probably don't want "process." everywhere
        options.isCfg = False
        return self.dumpPython(options)
    def getLeaves(self, leaves):
        self.topNode.getLeaves(leaves)
    def dumpPython(self, options):
        return "cms."+self.factory().__name__+"("+self.topNode.dumpPython(options)+")"


class _Sequence(_ModuleSeries):
    def factory(self):
        return cms.Sequence
    def type(self):
        return 'sequence'
class _Path(_ModuleSeries):
    def factory(self):
        return cms.Path
    def type(self):
        return 'path'
class _EndPath(_ModuleSeries):
    def factory(self):
        return cms.EndPath
    def type(self):
        return 'endpath'

    
class _MakeSeries(object):
    def __init__(self,factory):
        self.factory = factory
    def __call__(self,s,loc,toks):
        return (toks[0][0],self.factory(toks[0][1],s,loc,toks))
# really want either ! or -, not both
pathtoken = (pp.Combine(pp.Optional("!")+pp.Optional("-")+letterstart))|'&'|','|'('|')'
pathbody = pp.Group(letterstart+_equalTo
                    +_scopeBegin
                    +pp.Group(pp.OneOrMore(pathtoken)).setParseAction(_parsePathInReverse)
                    +_scopeEnd)
path = pp.Keyword('path').suppress()+pathbody.copy().setParseAction(_MakeSeries(_Path))
endpath = pp.Keyword('endpath').suppress()+pathbody.copy().setParseAction(_MakeSeries(_EndPath))
sequence = pp.Keyword('sequence').suppress()+pathbody.copy().setParseAction(_MakeSeries(_Sequence))


class _Schedule(object):
    """Stand-in for a Schedule since we can't build the real Schedule
    till the Paths have been created"""
    def __init__(self,labels):
        self.labels = labels
    def dumpPython(self, options):
        result = "cms.Schedule("
        # might have to add the word 'process' to each'
        newLabels = list()
        if options.isCfg:
            for label in self.labels:
                newLabels.append('process.'+label)
            result += ','.join(newLabels)
        result += ')\n'

        return result


def _makeSchedule(s,loc,toks):
    """create the appropriate parameter object from the tokens"""
    values = list(iter(toks[0][0]))
    p = _Schedule(values)
    return ('schedule',p)
    
schedule = pp.Group(pp.Keyword('schedule').suppress()+_equalTo+
                           _scopeBegin
                             +pp.Group(pp.Optional(pp.delimitedList(label)))
                           +_scopeEnd
                          ).setParseAction(_makeSchedule)

#==================================================================
# Other top level items
#==================================================================

class _ReplaceNode(object):
    """Handles the 'replace' command"""
    def __init__(self,path,setter,s,loc):
        global _fileStack
        self.path = path
        self.setter = setter
        self.forErrorMessage =(s,loc,_fileStack[-1])
        self.multiplesAllowed = setter.multiplesAllowed
    def getValue(self):
        return self.setter.value
    value = property(fget = getValue,
                     doc='returns the value of the replace command (for testing)')
    def rootLabel(self):
        return self.path[0]
    def do(self,process):
        if hasattr(self.setter, 'setProcess'):
            self.setter.setProcess(process)
        try:
            self._recurse(self.path,process)
        except Exception,e:
            raise pp.ParseException(self.forErrorMessage[0],
                                    self.forErrorMessage[1],
                                    "The replace statement '"+'.'.join(self.path)
                                    +"' had the error \n"+str(e)
                                    +"\n from file "+self.forErrorMessage[2]
                                    )
    def _setValue(self,obj,attr):
        self.setter.setValue(obj,attr)
    def _recurse(self,path,obj):
        if len(path) == 1:
            self._setValue(obj,path[0])
            return
        self._recurse(path[1:],getattr(obj,path[0]))
    def __repr__(self):
        options = PrintOptions()
        return self.dumpPython(options)
    def dumpPython(self, options):
        # translate true/false to True/False
        s = self.getValue()
        return '.'.join(self.path)+self.setter.dumpPython(options)

class _ReplaceSetter(object):
    """Used to 'set' an unknown type of value from a Replace node"""
    def __init__(self,value):
        self.value = value
        #one one replace of this type is allowed per configuration
        self.multiplesAllowed = False
    def setValue(self,obj,attr):
        theAt = getattr(obj,attr)
        #want to change the value, not the actual parameter
        #setattr(obj,attr,theAt._valueFromString(self.value).value())
        #by replacing the actual parameter we isolate ourselves from
        # 'replace' commands done by others on shared using blocks
        v=theAt._valueFromString(self.value)
        v.setIsTracked(theAt.isTracked())
        setattr(obj,attr,v)
    def dumpPython(self, options):
       # FIXME not really a repr, because of the = sign
       return " = "+self._pythonValue(self.value, options)
    @staticmethod
    def _pythonValue(value, options):
        #if it's a number, we don't want quotes
        result = str(value)
        nodots = result.replace('.','')
        if nodots.isdigit() or (len(nodots) > 0 and nodots[0] == '-') or (len(nodots) > 1 and (nodots[0:2] == '0x' or nodots[0:2] == '0X')):
            pass
        elif result == 'true':
            result = 'True'
        elif result == 'false':
            result = 'False'
        elif len(value) == 0:
            result = repr(value)
        elif result[0] == '[':
            l = eval(result)
            isVInputTag = False
            result = ''
            indented = False
            for i, x in enumerate(l):
                if i == 0:
                    if hasattr(x, "_nPerLine"):
                        nPerLine = x._nPerLine
                    else:
                        nPerLine = 5
                else:
                    result += ', '
                    if i % nPerLine == 0:
                        if not indented:
                            indented = True
                            options.indent()
                        result += '\n'+options.indentation()
                element = _ReplaceSetter._pythonValue(x, options) 
                if element.find('InputTag') != -1:
                   isVInputTag = True
                result += element
            if indented:
                options.unindent()
            if isVInputTag:
               result = "cms.VInputTag("+result+")"
            else:
               result = "["+result+"]"
        # some frontier address strings have colons and slashes 
        #elif result.find(':') != -1 and result.find('/') == -1:
        #    result = repr(cms.InputTag._valueFromString(result))
        else:
            # need the quotes
            result = repr(value)
        return result

 

class _ParameterReplaceSetter(_ReplaceSetter):
    """Base used to 'set' a PSet or VPSet replace node""" 
    def setValue(self,obj,attr):
        #need to preserve 'trackiness'
        theAt = getattr(obj,attr)
        self.value.setIsTracked(theAt.isTracked())
        setattr(obj,attr,self.value)
    @staticmethod
    def _pythonValue(value, options):
        return value.dumpPython(options)

class _VPSetReplaceSetter(_ParameterReplaceSetter):
    """Used to 'set' a VPSet replace node"""
    def __init__(self,value):
        super(_VPSetReplaceSetter,self).__init__(_makeVPSetFromList(value))
class _PSetReplaceSetter(_ParameterReplaceSetter):
    """Used to 'set' a VPSet replace node"""
    def __init__(self,value):
        super(_PSetReplaceSetter,self).__init__(_makePSetFromList(value))

class _SimpleListTypeExtendSetter(_ReplaceSetter):
    """replace command to extends a list"""
    def __init__(self,value):
        super(type(self),self).__init__(value)
        self.multiplesAllowed = True
    def setValue(self,obj,attr):
        theAt=getattr(obj,attr)
        theAt.extend(theAt._valueFromString(self.value))
    def dumpPython(self, options):
        return ".extend("+self._pythonValue(self.value, options)+")"


class _SimpleListTypeAppendSetter(_ReplaceSetter):
    """replace command to append to a list"""
    def __init__(self,value):
        super(type(self),self).__init__(value)
        self.multiplesAllowed = True
    def setValue(self,obj,attr):
        theAt=getattr(obj,attr)
        theAt.append(theAt._valueFromString([self.value])[0])
    def dumpPython(self, options):
        return ".append("+self._pythonValue(self.value, options)+")"



class _VPSetExtendSetter(_VPSetReplaceSetter):
    """replace command to extend a VPSet"""
    def __init__(self,value):
        super(type(self),self).__init__(value)
        self.multiplesAllowed = True
    def setValue(self,obj,attr):
        theAt=getattr(obj,attr)
        theAt.extend(self.value)
    def dumpPython(self, options):
        return ".extend("+self._pythonValue(self.value, options)+")"



class _VPSetAppendSetter(_PSetReplaceSetter):
    """replace command to append a PSet to a VPSet"""
    def __init__(self,value):
        super(type(self),self).__init__(value)
        self.multiplesAllowed = True
    def setValue(self,obj,attr):
        theAt=getattr(obj,attr)
        theAt.append(self.value)
    def dumpPython(self, options):
        return ".append("+self._pythonValue(self.value, options)+")"



class _IncrementFromVariableSetter(_ReplaceSetter):
    """replace command which gets its value from another parameter"""
    def __init__(self,value):
        self.valuePath = value
        super(type(self),self).__init__('.'.join(value))
        self.multiplesAllowed = True
        self.oldValue = None
    def setProcess(self,process):
        if self.oldValue is None:
            self.oldValue = self.value
            attr=None
            path = self.valuePath
            attr = process
            while path:
                attr = getattr(attr,path[0])
                path = path[1:]
            self.value = attr
    def setValue(self,obj,attr):
        theAt = getattr(obj,attr)
        #determine if the types are compatible
        try:
            if type(theAt) is type(self.value):
                theAt.extend(self.value)
            #see if theAt is a container and self.value can be added to it 
            else:
                theAt.append(self.value.value())
        except Exception, e:
            raise RuntimeError("replacing with "+self.oldValue+" failed because\n"+str(e))
    def dumpPython(self, options):
        v = str(self.value)
        if options.isCfg:
            v = 'process.'+v
        # assume variables ending in s are plural
        if v[0] == '[':
           return ".extend("+v+")"
        elif v.endswith('s'):
           return ".extend("+v+")"
        else:
           return ".append("+v+")"

        

class _MakeSetter(object):
    """Uses a 'factory' to create the proper Replace setter"""
    def __init__(self,setter):
        self.setter = setter
    def __call__(self,s,loc,toks):
        value = toks[0]
        if isinstance(value,pp.ParseResults):
            value = value[:]
        return self.setter(value)

def _makeReplace(s,loc,toks):
    try:
        path = toks[0][0]
        setter = toks[0][1]
        return ('.'.join(path),_ReplaceNode(list(path),setter,s,loc))
    except Exception, e:
        global _fileStack
        raise pp.ParseException(s,loc,"replace statement '"
                                +'.'.join(list(path))
                                +"' had the error \n"
                                +str(e)
                                +"\n from file "+_fileStack[-1])

_replaceValue = (pp.Group(_scopeBegin+_scopeEnd
                         ).setParseAction(_MakeSetter(_ReplaceSetter))|
                    (scopedParameters.copy()
                    ).setParseAction(_MakeSetter(_PSetReplaceSetter))|
                    (_scopeBegin+pp.Group(pp.delimitedList(scopedParameters))
                     +_scopeEnd).setParseAction(_MakeSetter(_VPSetReplaceSetter))|
                    (quotedString|
                     (_scopeBegin+pp.Group(pp.delimitedList(quotedString))+_scopeEnd)|
                     (_scopeBegin+pp.Group(pp.Optional(pp.delimitedList(any)))+_scopeEnd)|
                    any).setParseAction(_MakeSetter(_ReplaceSetter)))
_replaceExtendValue = (
                     scopedParameters.copy().setParseAction(_MakeSetter(_VPSetAppendSetter)) |
                     (_scopeBegin+pp.Group(pp.delimitedList(scopedParameters))
                      +_scopeEnd).setParseAction(_MakeSetter(_VPSetExtendSetter))|
                     ((_scopeBegin+pp.Group(pp.delimitedList(quotedString))+_scopeEnd)|
                      (_scopeBegin+pp.Group(pp.Optional(pp.delimitedList(any)))+_scopeEnd)
                     ).setParseAction(_MakeSetter(_SimpleListTypeExtendSetter)) |
                     (pp.Group(letterstart+pp.OneOrMore(pp.Literal('.').suppress()+letterstart)).setParseAction(
                        _MakeSetter(_IncrementFromVariableSetter))) |
                     ((quotedString|any).setParseAction(_MakeSetter(_SimpleListTypeAppendSetter))) 
                  )
_plusEqualTo = pp.Suppress('+=')
#NOTE: can't use '_equalTo' since it checks for a 'valid' label and gets confused
# when += appears
_eqTo = pp.Suppress("=")
replace = pp.Group(pp.Keyword('replace').suppress()+
                   pp.Group(letterstart+
                            pp.OneOrMore(pp.Literal('.').suppress()+letterstart)
                            )+
                   ((_plusEqualTo+_replaceExtendValue
                    ) | (
                    _eqTo+_replaceValue))
                  ).setParseAction(_makeReplace) 

class _ProcessAdapter(object):
    def __init__(self,seqs,process):
        self.__dict__['_seqs'] = seqs
        self.__dict__['_process'] = process
    def seqs(self):
        return self.__dict__['_seqs']
    def process(self):
        return self.__dict__['_process']
    def __getattr__(self,name):
        if hasattr(self.process(), name):
            return getattr(self.process(),name)
        setattr(self.process(),name,self.seqs()[name].make(self))
        return getattr(self.process(),name)
    def __setattr__(self,name,value):
        if hasattr(self.process(),name):
            return
        setattr(self.process(),name,value)
def _finalizeProcessFragment(values,usingLabels):
    try:
        values = _validateLabelledList(values)
        values = _findAndHandleProcessBlockIncludes(values)
        values = _validateLabelledList(values)
    except Exception, e:
        raise RuntimeError("the configuration contains the error \n"+str(e))
    #now deal with series
    d = SortedKeysDict(values)
    dct = dict(d)
    replaces=[]
    sequences = {}
    series = []
    for label,item in values:
        if isinstance(item,_ReplaceNode):
            replaces.append(item)
            #replace statements are allowed to have multiple identical labels
            if label in d:
                del d[label]
            if label in dct:
                del dct[label]
        elif isinstance(item,_Sequence):
            sequences[label]=item
            del dct[label]
        elif isinstance(item,_ModuleSeries):
            series.append((label,item))
            del dct[label]
    try:
        #pset replaces must be done first since PSets can be used in a 'using'
        # statement so we want their changes to be reflected
        adapted = _DictAdapter(dict(d),True)
        #what order do we process replace and using directives?
        # running a test on the C++ cfg parser it appears replace
        # always happens before using
        for replace in replaces:
            if isinstance(getattr(adapted,replace.rootLabel()),cms.PSet):
                replace.do(adapted)
        _findAndHandleProcessUsingBlock(values)
        for replace in replaces:
            if not isinstance(getattr(adapted,replace.rootLabel()),cms.PSet):
                replace.do(adapted)
        # maybe the user said something like 'replace a.b = {using bl}
        _findAndHandleProcessUsingBlock(values)
    except Exception, e:
        raise RuntimeError("the configuration contains the error \n"+str(e))    
    #FIX: now need to create Sequences, Paths, EndPaths from the available
    # information
    #now we don't want 'source' to be added to 'd' but we do not want
    # copies either
    adapted = _DictAdapter(d)
    pa = _ProcessAdapter(sequences,_DictAdapter(dct))
    for label,obj in sequences.iteritems():
        if label not in dct:
            d[label]=obj.make(pa)
            dct[label]=d[label]
        else:
            d[label] = dct[label]
    for label,obj in series:
        d[label]=obj.make(adapted)
    return d
#==================================================================
# Process
#==================================================================
def _getCompressedNodes(s,loc, values):
    """Inlines the using statements, but not the Includes or Replaces"""
    compressedValues = []
    for l,v in values:
        compressedValues.append((l,v))

    try:
        compressedValues = _validateLabelledList(compressedValues)
        expandedValues = _findAndHandleProcessBlockIncludes(compressedValues)
        expandedValues = _validateLabelledList(expandedValues)
        #_findAndHandleProcessUsingBlock(expandedValues)
    except Exception, e:
        raise pp.ParseFatalException(s,loc,"the process contains the error \n"+str(e))

    return compressedValues

def _dumpCfg(s,loc,toks):
    label = toks[0][0]
    p=cms.Process(label)

    values = _getCompressedNodes(s, loc, list(iter(toks[0][1])) )
    options = PrintOptions()
    options.isCfg = True
    result = "import FWCore.ParameterSet.Config as cms\n\nprocess = cms.Process(\""+label+"\")\n"
    result += dumpPython(values, options)
    return result

def _makeProcess(s,loc,toks):
    """create a Process from the tokens"""
    #print toks
    label = toks[0][0]
    p=cms.Process(label)
    values = list(iter(toks[0][1]))
    try:
        values = _validateLabelledList(values)
        values = _findAndHandleProcessBlockIncludes(values)
        values = _validateLabelledList(values)
    except Exception, e:
        raise RuntimeError("the process contains the error \n"+str(e)
                          )


    #now deal with series
    d = dict(values)
    sequences={}
    series=[] #order matters for a series
    replaces=[]
    prefers = {}
    schedule = None


    #sequences must be added before path or endpaths
    #sequences may contain other sequences so we need to do recursive construction
    try:
        for label,item in values:
            if isinstance(item,_Sequence):
                sequences[label]=item
                del d[label]
            elif isinstance(item,_ModuleSeries):
                series.append((label,item))
                del d[label]
            elif isinstance(item,_ReplaceNode):
                replaces.append(item)
                #replace statements are allowed to have multiple identical labels
                if label in d: del d[label]
            elif isinstance(item,_Schedule):
                if schedule is None:
                    schedule = item
                    del d[label]
                else:
                    raise RuntimeError("multiple 'schedule's are present, only one is allowed")
            elif isinstance(item,cms.ESPrefer):
                prefers[label[0:-7]]=item
                del d[label]
        #pset replaces must be done first since PSets can be used in a 'using'
        # statement so we want their changes to be reflected
        adapted = _DictCopyAdapter(d)
        #what order do we process replace and using directives?
        # running a test on the C++ cfg parser it appears replace
        # always happens before using
        for replace in replaces:
            if isinstance(getattr(adapted,replace.rootLabel()),cms.PSet):
                replace.do(adapted)
        _findAndHandleProcessUsingBlock(values)
        for replace in replaces:
            if not isinstance(getattr(adapted,replace.rootLabel()),cms.PSet):
                replace.do(adapted)
        #NEED to call this a second time so replace statements applying to modules
        # where the replace statements contain using statements will have the
        # using statements replaced by their actual values
        _findAndHandleProcessUsingBlock(values)


        # adding modules to the process involves cloning.
        # but for the usings we only know the original object
        # so we do have to keep a lookuptable
        # FIXME  <- !!
        global _lookuptable
        _lookuptable = {}
        
        for label,obj in d.iteritems():
            setattr(p,label,obj)
            if not isinstance(obj,list): _lookuptable[obj] = label
        for label,obj in prefers.iteritems():
            setattr(p,label,obj)
        pa = _ProcessAdapter(sequences,p)
        for label,obj in sequences.iteritems():
            setattr(pa,label,obj.make(pa))
        for label,obj in series:
            setattr(p,label,obj.make(p))
        if schedule is not None:
            pathlist = []
            for label in schedule.labels:
               pathlist.append( getattr(p,label))
            p.schedule = cms.Schedule(*pathlist)
    except Exception, e:
        raise RuntimeError("the process contains the error \n"+str(e))    
#    p = cms.PSet(*[],**d)
    return p


processNode = plugin|PSetParameter|VPSetParameter|block|include|path|endpath|sequence|schedule|replace
processBody = pp.OneOrMore(processNode)
processBody.ignore(pp.cppStyleComment)
processBody.ignore(pp.pythonStyleComment)


#.cfi
onlyPlugin = plugin|pp.empty+pp.StringEnd()
#.cff
onlyProcessBody = processBody|pp.empty+pp.StringEnd()
onlyProcessBody.ignore(pp.cppStyleComment)
onlyProcessBody.ignore(pp.pythonStyleComment)
onlyParameters = parameters|pp.empty+pp.StringEnd()
onlyFragment =processBody|parameters|plugin|pp.empty+pp.StringEnd()
onlyFragment.ignore(pp.cppStyleComment)
onlyFragment.ignore(pp.pythonStyleComment)
#.cfg
process = pp.Group(pp.Suppress('process')+label+_equalTo+
                   _scopeBegin+
                     pp.Group(processBody)+
                   _scopeEnd).setParseAction(_makeProcess)+pp.StringEnd()
process.ignore(pp.cppStyleComment)
process.ignore(pp.pythonStyleComment)

cfgDumper = pp.Group(pp.Suppress('process')+label+_equalTo+
                   _scopeBegin+
                     pp.Group(processBody)+
                   _scopeEnd).setParseAction(_dumpCfg)+pp.StringEnd()
cfgDumper.ignore(pp.cppStyleComment)
cfgDumper.ignore(pp.pythonStyleComment)


def dumpDict(d):
    result = 'import FWCore.ParameterSet.Config as cms\n\n'
    options = PrintOptions()
    options.isCfg = False
    return result+dumpPython(d, options)


def dumpPython(d, options):
    # play it safe: includes first, then others, then replaces
    includes = ''
    replaces = ''
    others = ''
    sequences = ''
    schedules = ''
    prefix = ''
    if options.isCfg:
        prefix = 'process.'
    for key,value in d:
        newLabel = prefix+key
        # dashes aren't allowed in python identifier
        newLabel.replace('-','_')
        if isinstance(value,_IncludeFromNode):
            value.createFile(False)
            includes += value.dumpPythonAs(newLabel, options)
        elif isinstance(value,_IncludeNode):
            value.createFile(False)
            includes += value.dumpPython(options)+"\n"
        elif isinstance(value,_ReplaceNode):
            replaces += prefix+value.dumpPython(options)+"\n"
        elif isinstance(value,_ModuleSeries):
            sequences += newLabel+" = "+value.dumpPython(options)+"\n"
        elif isinstance(value,_Schedule):
            schedules += newLabel+" = "+value.dumpPython(options)+"\n"
        elif isinstance(value, cms.ESPrefer):
            others += value.dumpPythonAs(key, options)
        else:
            others += newLabel+" = "+value.dumpPython(options)+"\n"
    return includes+others+sequences+schedules+replaces+"\n"

class _ConfigReturn(object):
    def __init__(self,d):
        for key,value in d.iteritems():
            setattr(self, key, value)
    def commentedOutRepr(self):
        # make sure all the top-level Labelables are labelled
        for key,value in self.__dict__.iteritems():
            if isinstance(value, cms._Labelable):
                value.setLabel(key)
        result = 'import FWCore.ParameterSet.Config as cms\n'
        # play it safe: includes first, then others, then replaces
        return dumpDict(self.__dict__.iteritems())

def parseCfgFile(fileName):
    """Read a .cfg file and create a Process object"""
    #NOTE: should check for file first in local directory
    # and then using FileInPath

    global _allUsingLabels
    global _fileStack
    _allUsingLabels = set()
    oldFileStack = _fileStack
    _fileStack = [fileName]
    import os.path
    try:
        if os.path.exists(fileName):
            f=open(fileName)
        else:
            f=_fileFactory(fileName)
        return process.parseFile(f)[0]
    finally:
        _fileStack = oldFileStack

def parseCffFile(fileName):
    """Read a .cff file and return a dictionary"""
    global _fileStack
    _fileStack.append(fileName)
    try:
        t=onlyFragment.parseFile(_fileFactory(fileName))
        global _allUsingLabels
        #_allUsingLabels = set() # do I need to reset here?
        d=_finalizeProcessFragment(t,_allUsingLabels)
        return _ConfigReturn(d)
    finally:
        _fileStack.pop()

def parseConfigString(aString):
    """Read an old style config string and return a dictionary"""
    t=onlyFragment.parseString(aString)
    global _allUsingLabels
    d=_finalizeProcessFragment(t,_allUsingLabels)
    return _ConfigReturn(d)
                    
def dumpCfg(fileName):
    return cfgDumper.parseFile(_fileFactory(fileName))[0]


def dumpCff(fileName):
    # we need to process the Usings, but leave the Includes and Replaces
    values = onlyFragment.parseFile(_fileFactory(fileName))
    # copy from whatever got returned into a list
    compressedValues = _getCompressedNodes(fileName, 0, values)
    # make sure all the top-level Labelables are labelled
    for key,value in compressedValues:
        if isinstance(value, cms._Labelable):
            value.setLabel(key)

    _validateSequences(compressedValues)

    return dumpDict(compressedValues)

def _validateSequences(values):
    """ Takes the usual list of pairs, and for each
    sequence, checks see if the keys are defined yet.
    Adds a placeholders for missing keys"""
    keys = set()
    leaves = list()
    expandedValues = _findAndHandleProcessBlockIncludes(values)
    for key, value in expandedValues:
        keys.add(key)
        if isinstance(value, _ModuleSeries):
           value.getLeaves(leaves)
    for leaf in leaves:
        if not leaf._label in keys:
            leaf._label = "cms.SequencePlaceholder(\""+leaf._label+"\")"

def processFromString(configString):
    """Reads a string containing the equivalent content of a .cfg file and
    creates a Process object"""
    global _allUsingLabels
    global _fileStack
    _allUsingLabels = set()
    oldFileStack = _fileStack
    _fileStack = ["{string}"]
    try:
        return process.parseString(configString)[0]
    finally:
        _fileStack = oldFileStack

def importConfig(fileName):
    """Use the file extension to decide how to parse the file"""
    ext = fileName[fileName.rfind('.'):]
    if ext == '.cfg':
        return parseCfgFile(fileName)
    if ext != '.cff' and ext != '.cfi':
        raise RuntimeError("the file '"+fileName+"' has an unknown extension")
    return parseCffFile(fileName)
    
if __name__=="__main__":
    import unittest
    import StringIO
    class TestFactory(object):
        def __init__(self,name, contents):
            self._name=name

            self._contents = contents
        def __call__(self, filename):
            if self._name != filename:
                raise RuntimeError("wrong file name, expected "+self._name+' saw '+filename)
            return StringIO.StringIO(self._contents)

    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            #print 'testing'
        def testPlaceholder(self):
            t=path.parseString('path p = {out}')
            _validateSequences(t)
            self.assertEqual(repr(t[0][1]), "cms.Path(cms.SequencePlaceholder(\"out\"))")

        def testLetterstart(self):
            t = letterstart.parseString("abcd")
            self.assertEqual(len(t),1)
            self.assertEqual(t[0],"abcd")
            t = letterstart.parseString("a1cd")
            self.assertEqual(len(t),1)
            self.assertEqual(t[0],"a1cd")
            t = letterstart.parseString("a_cd")
            self.assertEqual(t[0],"a_cd")
            t = letterstart.parseString("a-cd")
            self.assertEqual(t[0],"a-cd")
            self.assertRaises(pp.ParseBaseException,letterstart.parseString,("1abc"))
        def testParameters(self):
            t=onlyParameters.parseString("bool blah = True")
            d =dict(iter(t))
            self.assertEqual(type(d['blah']),cms.bool)
            self.assertEqual(d['blah'].value(),True)
            t=onlyParameters.parseString("bool blah = 1")
            d =dict(iter(t))
            self.assertEqual(type(d['blah']),cms.bool)
            self.assertEqual(d['blah'].value(),True)
            t=onlyParameters.parseString("bool blah = False")
            d =dict(iter(t))
            self.assertEqual(type(d['blah']),cms.bool)
            self.assertEqual(d['blah'].value(),False)
            t=onlyParameters.parseString("bool blah = 2")
            d =dict(iter(t))
            self.assertEqual(type(d['blah']),cms.bool)
            self.assertEqual(d['blah'].value(),True)
            t=onlyParameters.parseString("vint32 blah = {}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vint32)
            self.assertEqual(len(d['blah']),0)
            t=onlyParameters.parseString("vint32 blah = {1}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vint32)
            self.assertEqual(d['blah'],[1])
            t=onlyParameters.parseString("vint32 blah = {1,2}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vint32)
            self.assertEqual(d['blah'],[1,2])
            t=onlyParameters.parseString("string blah = 'a string'")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.string)
            self.assertEqual(d['blah'].value(),'a string')
            t=onlyParameters.parseString('string blah = "a string"')
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.string)
            self.assertEqual(d['blah'].value(),'a string')
            t=onlyParameters.parseString('string blah = "\\0"')
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.string)
            self.assertEqual(d['blah'].value(),'\0')

            t=onlyParameters.parseString("vstring blah = {}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vstring)
            self.assertEqual(len(d['blah']),0)
            t=onlyParameters.parseString("vstring blah = {'abc'}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vstring)
            self.assertEqual(d['blah'],['abc'])
            t=onlyParameters.parseString("vstring blah = {'abc','def'}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.vstring)
            self.assertEqual(d['blah'],['abc','def'])
            
            t = onlyParameters.parseString("InputTag blah = tag")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.InputTag)
            self.assertEqual(d['blah'].moduleLabel,'tag')
            self.assertEqual(d['blah'].productInstanceLabel,'')
            t = onlyParameters.parseString("InputTag blah = tag:")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.InputTag)
            self.assertEqual(d['blah'].moduleLabel,'tag')
            self.assertEqual(d['blah'].productInstanceLabel,'')

            t = onlyParameters.parseString("InputTag blah =tag:youIt")
            # FAILS!
            #  = onlyParameters.parseString("InputTag blah = \"tag:youIt\"")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.InputTag)
            self.assertEqual(d['blah'].moduleLabel,'tag')
            self.assertEqual(d['blah'].productInstanceLabel,'youIt')

            t = onlyParameters.parseString("InputTag blah = tag::proc")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.InputTag)
            self.assertEqual(d['blah'].moduleLabel,'tag')
            self.assertEqual(d['blah'].processName,'proc')
                                                 
            t = onlyParameters.parseString("InputTag blah = tag:youIt:Now")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.InputTag)
            self.assertEqual(d['blah'].moduleLabel,'tag')
            self.assertEqual(d['blah'].productInstanceLabel,'youIt')
            self.assertEqual(d['blah'].processName,'Now')

            t=onlyParameters.parseString("VInputTag blah = {}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VInputTag)
            self.assertEqual(len(d['blah']),0)
            t=onlyParameters.parseString("VInputTag blah = {abc}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VInputTag)
            self.assertEqual(d['blah'],[cms.InputTag('abc')])
            t=onlyParameters.parseString("VInputTag blah = {abc, def}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VInputTag)
            self.assertEqual(d['blah'],[cms.InputTag('abc'),cms.InputTag('def')])
            t=onlyParameters.parseString("VInputTag blah = {'abc', def}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VInputTag)
            self.assertEqual(d['blah'],[cms.InputTag('abc'),cms.InputTag('def')])


            t=onlyParameters.parseString("EventID eid= 1:2")
            d=dict(iter(t))
            self.assertEqual(type(d['eid']),cms.EventID)
            self.assertEqual(d['eid'].run(), 1)
            self.assertEqual(d['eid'].event(), 2)
            t=onlyParameters.parseString("LuminosityBlockID lbid= 1:2")
            d=dict(iter(t))
            self.assertEqual(type(d['lbid']),cms.LuminosityBlockID)
            self.assertEqual(d['lbid'].run(), 1)
            self.assertEqual(d['lbid'].luminosityBlock(), 2)


            
            t=onlyParameters.parseString("PSet blah = {}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.PSet)

            t=onlyParameters.parseString("PSet blah = {int32 ick = 1 }")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.PSet)
            self.assertEqual(d['blah'].ick.value(), 1)            

            t=onlyParameters.parseString("""PSet blah = {
                                         int32 ick = 1
                                         }""")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.PSet)
            self.assertEqual(d['blah'].ick.value(), 1)            

            t=onlyParameters.parseString("""PSet blah = {
                                         InputTag t1 = abc: 
                                         InputTag t2 = def:GHI
                                         }""")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.PSet)

            t=onlyParameters.parseString("VPSet blah = {}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VPSet)
            
            t=onlyParameters.parseString("VPSet blah = { {} }")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VPSet)
            self.assertEqual(len(d['blah']),1)

            t=onlyParameters.parseString("VPSet blah = { {int32 ick = 1 } }")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VPSet)
            self.assertEqual(len(d['blah']),1)
            t=onlyParameters.parseString("VPSet blah = { {int32 ick = 1 }, {int32 ick =2} }")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.VPSet)
            self.assertEqual(len(d['blah']),2)
            
            t=onlyParameters.parseString("secsource blah = Foo {int32 ick=1}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.SecSource)

        def testValidation(self):
            self.assertRaises(pp.ParseFatalException,onlyParameters.parseString,("""
PSet blah = {
  int32 ick = 1
  int32 ick = 2
}"""),**dict())
            global _fileFactory
            oldFactory = _fileFactory
            try:
                _fileFactory = TestFactory('blah.cfi', 'int32 blah = 1')
                t=onlyParameters.parseString("""
PSet blah = {
  include "blah.cfi"
  include "blah.cfi"
}""")
                d = dict(iter(t))
                self.assertEqual(getattr(d['blah'],"blah").value(), 1)
                _fileFactory = TestFactory('blah.cfi', 'int32 blah = 1')
                self.assertRaises(pp.ParseFatalException,onlyParameters.parseString,("""
PSet blah = {
  include "blah.cfi"
   int32 blah = 2
}"""),**dict())
            finally:
                _fileFactory=oldFactory
        def testUsing(self):
            #self.assertRaises(pp.ParseFatalException,onlyParameters.parseString,("PSet blah = {using ick}"),**dict())
            t=onlyParameters.parseString("PSet blah = {using ick}")
            d=dict(iter(t))
            self.assertEqual(type(d['blah']),cms.PSet)
            self.assertEqual(d['blah'].using_ick.value(), 'ick')
        def testInclude(self):
            #for testing use a different factory
            import StringIO
            global _fileFactory
            oldFactory = _fileFactory
            try:
                _fileFactory = TestFactory('Sub/Pack/data/foo.cff', 'int32 blah = 1')
                t=onlyParameters.parseString("PSet blah = {include 'Sub/Pack/data/foo.cff'}")
                d=dict(iter(t))
                self.assertEqual(type(d['blah']),cms.PSet)
                self.assertEqual(getattr(d['blah'],"blah").value(), 1)
                
                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi', 'module foo = TestProd {}')
                t=onlyProcessBody.parseString("include 'Sub/Pack/data/foo.cfi'")
                d=dict(iter(t))
                self.assertEqual(d['Sub/Pack/data/foo.cfi'].filename, 'Sub/Pack/data/foo.cfi')
                t = _findAndHandleProcessBlockIncludes(t)
                d=dict(iter(t))
                self.assertEqual(type(d['foo']),cms.EDProducer)
                
                #test ending with a comment
                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi', """module c = CProd {}
#""")
                t=onlyProcessBody.parseString("""module b = BProd {}
                                              include 'Sub/Pack/data/foo.cfi'""")
                d=dict(iter(t))
                self.assertEqual(d['Sub/Pack/data/foo.cfi'].filename, 'Sub/Pack/data/foo.cfi')
                t = _findAndHandleProcessBlockIncludes(t)

                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi', "include 'Sub/Pack/data/foo.cfi'")
                t=onlyProcessBody.parseString("include 'Sub/Pack/data/foo.cfi'")
                d=dict(iter(t))
                self.assertEqual(d['Sub/Pack/data/foo.cfi'].filename, 'Sub/Pack/data/foo.cfi')
                self.assertRaises(RuntimeError,_findAndHandleProcessBlockIncludes,t)
                #t = _findAndHandleProcessBlockIncludes(t)

                _fileFactory = TestFactory('Sub/Pack/data/foo.cff', '#an empty file')
                t=onlyParameters.parseString("PSet blah = {include 'Sub/Pack/data/foo.cff'}")
                d=dict(iter(t))
                self.assertEqual(type(d['blah']),cms.PSet)

                #test block
                _fileFactory = TestFactory('Sub/Pack/data/foo.cff', """block c = { ##
                                           ##
                                           double EBs25notContainment = 0.965}
""")
                t=onlyProcessBody.parseString("""module b = BProd {using c}
                                              include 'Sub/Pack/data/foo.cff'""")
                d=dict(iter(t))
                self.assertEqual(d['Sub/Pack/data/foo.cff'].filename, 'Sub/Pack/data/foo.cff')
                t = _findAndHandleProcessBlockIncludes(t)
                
                _fileFactory = TestFactory('Sub/Pack/data/foo.cff',
                                           """path p = {doesNotExist}""")
                try:
                    process.parseString("""process T = { include 'Sub/Pack/data/foo.cff' }""")
                except Exception,e:
                    self.assertEqual(str(e),
"""the process contains the error 
path 'p' contains the error: 'Process' object has no attribute 'doesNotExist'
 from file Sub/Pack/data/foo.cff (at char 4), (line:1, col:5)""")
                else:
                    self.fail("failed to throw exception")
            finally:
                _fileFactory = oldFactory
        def testParseCffFile(self):
            import StringIO
            global _fileFactory
            oldFactory = _fileFactory
            try:
                _fileFactory = TestFactory('Sub/Pack/data/foo.cff',
                                           """module a = AProducer {}
                                           sequence s1 = {s}
                                           sequence s = {a}""")
                p=parseCffFile('Sub/Pack/data/foo.cff')
                self.assertEqual(p.a.type_(),'AProducer')
                self.assertEqual(type(p.s1),cms.Sequence)
                self.assertTrue(p.s1._seq is p.s)
                pr=cms.Process('Test')
                pr.extend(p)
                self.assertEqual(str(pr.s),'a')
                pr.dumpConfig()
            finally:
                _fileFactory = oldFactory
            
        def testPlugin(self):
            t=plugin.parseString("source = PoolSource { }")
            d=dict(iter(t))
            self.assertEqual(type(d['source']),cms.Source)
            self.assertEqual(d['source'].type_(),"PoolSource")
            
            t=plugin.parseString("source = PoolSource { }")
            d=dict(iter(t))
            self.assertEqual(type(d['source']),cms.Source)
            self.assertEqual(d['source'].type_(),"PoolSource")

            t=plugin.parseString("service = MessageLogger { }")
            d=dict(iter(t))
            self.assertEqual(type(d['MessageLogger']),cms.Service)
            self.assertEqual(d['MessageLogger'].type_(),"MessageLogger")
            
            t=plugin.parseString("module foo = FooMod { }")
            d=dict(iter(t))
            self.assertEqual(type(d['foo']),cms.EDFilter)
            self.assertEqual(d['foo'].type_(),"FooMod")
            
            t=plugin.parseString("module out = AsciiOutputModule { }")
            d=dict(iter(t))
            self.assertEqual(type(d['out']),cms.OutputModule)
            self.assertEqual(d['out'].type_(),"AsciiOutputModule")

            t=plugin.parseString("module foo = FooProd { }")
            d=dict(iter(t))
            self.assertEqual(type(d['foo']),cms.EDProducer)
            self.assertEqual(d['foo'].type_(),"FooProd")
            
            t=plugin.parseString("module foo = FooProducer { }")
            d=dict(iter(t))
            self.assertEqual(type(d['foo']),cms.EDProducer)
            self.assertEqual(d['foo'].type_(),"FooProducer")

            t=plugin.parseString("es_module = NoLabel {}")
            d=dict(iter(t))
            self.assertEqual(type(d['NoLabel']),cms.ESProducer)
            self.assertEqual(d['NoLabel'].type_(),"NoLabel")
 
            t=plugin.parseString("es_module foo = WithLabel {}")
            d=dict(iter(t))
            self.assertEqual(type(d['foo']),cms.ESProducer)
            self.assertEqual(d['foo'].type_(),"WithLabel")
            
            t=plugin.parseString("""module mix = MixingModule {
            secsource input = PoolRASource {
               vstring fileNames = {}
               }
            }
            """)
            global _fileFactory
            oldFactory = _fileFactory
            try:
                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi', 'module foo = TestProd {}')
                t=plugin.parseString("module bar = foo from 'Sub/Pack/data/foo.cfi'")
                t = _findAndHandleProcessBlockIncludes(t)
                d = dict(iter(t))
                self.assertEqual(type(d['bar']),cms.EDProducer)
                self.assertEqual(d['bar'].type_(),"TestProd")
                
                
                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi', 'es_module foo = TestProd {}')
                t=plugin.parseString("es_module bar = foo from 'Sub/Pack/data/foo.cfi'")
                t = _findAndHandleProcessBlockIncludes(t)
                d=dict(iter(t))
                self.assertEqual(type(d['bar']),cms.ESProducer)
                self.assertEqual(d['bar'].type_(),"TestProd")
            finally:
                _fileFactory = oldFactory

            self.assertRaises(pp.ParseFatalException,
                              plugin.parseString,
                              ("""es_module foo = WithLabel {
                                 uint32 a = 1
                                 int32 a = 1
                                 }"""))
        def testProcess(self):
            global _allUsingLabels
            _allUsingLabels = set()
            t=process.parseString(
"""
process RECO = {
   source = PoolSource {
     untracked vstring fileNames = {"file:foo.root"}
   }
   module out = PoolOutputModule {
     untracked string fileName = "blah.root"
   }
   path p = {out}
}""")
            self.assertEqual(t[0].source.type_(),"PoolSource")
            self.assertEqual(t[0].out.type_(),"PoolOutputModule")
            self.assertEqual(type(t[0].p),cms.Path)
            self.assertEqual(str(t[0].p),'out')
            #print t[0].dumpConfig()
            import StringIO
            global _fileFactory
            oldFactory = _fileFactory
            try:
                _fileFactory = TestFactory('Sub/Pack/data/foo.cfi',
                                           'module foo = FooProd {}')
                _allUsingLabels = set()
                t=process.parseString(
"""
process RECO = {
   source = PoolSource {
     untracked vstring fileNames = {"file:foo.root"}
   }
   include "Sub/Pack/data/foo.cfi"
   path p = {foo}
}""")
                self.assertEqual(t[0].foo.type_(),"FooProd")
            finally:
                _fileFactory = oldFactory

            _allUsingLabels = set()
            t=process.parseString(
"""
process RECO = {
   source = PoolSource {
     untracked vstring fileNames = {"file:foo.root"}
   }
   module out = PoolOutputModule {
     untracked string fileName = "blah.root"
   }
   endpath e = {out}
   module foo = FooProd {}
   module bar = BarProd {}
   module fii = FiiProd {}
   path p = {s&fii}
   sequence s = {foo,bar}
}""")
            self.assertEqual(str(t[0].p),'foo*bar+fii')
            self.assertEqual(str(t[0].s),'foo*bar')
            t[0].dumpConfig()

            _allUsingLabels = set()
            t=process.parseString(
"""
process RECO = {
   source = PoolSource {
     untracked vstring fileNames = {"file:foo.root"}
   }
   module out = PoolOutputModule {
     untracked string fileName = "blah.root"
   }
   endpath e = {out}
   module foo = FooProd {}
   module bar = BarProd {}
   module fii = FiiProd {}
   path p = {s&!fii}
   sequence s = {foo,bar}
}""")
            self.assertEqual(str(t[0].p),'foo*bar+~fii')
            self.assertEqual(str(t[0].s),'foo*bar')
            t[0].dumpConfig()
            
            s="""
process RECO = {
    module foo = FooProd {}
    path p = {fo}
}
"""
            self.assertRaises(RuntimeError,process.parseString,(s),**dict())
            try:
                _allUsingLabels = set()
                t=process.parseString(s)
            except Exception, e:
                print e

            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep = {
      vstring outputCommands = {"keep blah_*_*_*"}
   }
   replace outputStuff.outputCommands += toKeep.outputCommands
}
""")
            self.assertEqual(t[0].outputStuff.outputCommands,["drop *","keep blah_*_*_*"])

            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
  module i = iterativeCone5CaloJets from "FWCore/ParameterSet/test/chainIncludeModule.cfi"
}
""")
            self.assertEqual(t[0].i.jetType.value(), "CaloJet")


            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
  include "FWCore/ParameterSet/test/chainIncludeBlock.cfi"
  replace CaloJetParameters.inputEMin = 0.1
  module i = iterativeConeNoBlock from "FWCore/ParameterSet/test/chainIncludeModule2.cfi"
}
""")
            self.assertEqual(t[0].i.jetType.value(), "CaloJet")
            self.assertEqual(t[0].i.inputEMin.value(), 0.1)



            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep1 = {
      vstring outputCommands = {"keep blah1_*_*_*"}
   }
   block toKeep2 = {
      vstring outputCommands = {"keep blah2_*_*_*"}
   }
   block toKeep3 = {
      vstring outputCommands = {"keep blah3_*_*_*"}
   }
   replace outputStuff.outputCommands += toKeep1.outputCommands
   replace outputStuff.outputCommands += toKeep2.outputCommands
   replace outputStuff.outputCommands += toKeep3.outputCommands
}
""")
            self.assertEqual(t[0].outputStuff.outputCommands,["drop *",
                                                              "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*"])

            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep1 = {
      vstring outputCommands = {"keep blah1_*_*_*"}
   }
   block toKeep2 = {
      vstring outputCommands = {"keep blah2_*_*_*"}
   }
   block toKeep3 = {
      vstring outputCommands = {"keep blah3_*_*_*"}
   }
   replace outputStuff.outputCommands += toKeep1.outputCommands
   replace outputStuff.outputCommands += toKeep2.outputCommands
   replace outputStuff.outputCommands += toKeep3.outputCommands

    module out = PoolOutputModule {
        using outputStuff
    }
}
""")
            self.assertEqual(t[0].out.outputCommands,["drop *",
                                                              "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*"])
            t=process.parseString("""
process RECO = {
   block FEVTEventContent = {
      vstring outputCommands = {"drop *"}
   }
   block FEVTSIMEventContent = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep1 = {
      vstring outputCommands = {"keep blah1_*_*_*"}
   }
   block toKeep2 = {
      vstring outputCommands = {"keep blah2_*_*_*"}
   }
   block toKeep3 = {
      vstring outputCommands = {"keep blah3_*_*_*"}
   }
   
   block toKeepSim1 = {
      vstring outputCommands = {"keep blahs1_*_*_*"}
   }
   block toKeepSim2 = {
      vstring outputCommands = {"keep blahs2_*_*_*"}
   }
   block toKeepSim3 = {
      vstring outputCommands = {"keep blahs3_*_*_*"}
   }
   
   replace FEVTEventContent.outputCommands += toKeep1.outputCommands
   replace FEVTEventContent.outputCommands += toKeep2.outputCommands
   replace FEVTEventContent.outputCommands += toKeep3.outputCommands

   replace FEVTSIMEventContent.outputCommands += FEVTEventContent.outputCommands

   replace FEVTSIMEventContent.outputCommands += toKeepSim1.outputCommands
   replace FEVTSIMEventContent.outputCommands += toKeepSim2.outputCommands
   replace FEVTSIMEventContent.outputCommands += toKeepSim3.outputCommands

}
""")
            self.assertEqual(t[0].FEVTEventContent.outputCommands,["drop *",
                                                              "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*"])

            self.assertEqual(t[0].FEVTSIMEventContent.outputCommands,["drop *",
                                                                      "drop *",
                                                            "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*",
                                                            "keep blahs1_*_*_*",
                                                              "keep blahs2_*_*_*",
                                                              "keep blahs3_*_*_*"])

            t=process.parseString("""
process RECO = {
   block FEVTEventContent = {
      vstring outputCommands = {"drop *"}
   }
   block FEVTSIMEventContent = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep1 = {
      vstring outputCommands = {"keep blah1_*_*_*"}
   }
   block toKeep2 = {
      vstring outputCommands = {"keep blah2_*_*_*"}
   }
   block toKeep3 = {
      vstring outputCommands = {"keep blah3_*_*_*"}
   }
   
   block toKeepSim1 = {
      vstring outputCommands = {"keep blahs1_*_*_*"}
   }
   block toKeepSim2 = {
      vstring outputCommands = {"keep blahs2_*_*_*"}
   }
   block toKeepSim3 = {
      vstring outputCommands = {"keep blahs3_*_*_*"}
   }
   
   replace FEVTEventContent.outputCommands += toKeep1.outputCommands
   replace FEVTEventContent.outputCommands += toKeep2.outputCommands
   replace FEVTEventContent.outputCommands += toKeep3.outputCommands

   replace FEVTSIMEventContent.outputCommands += FEVTEventContent.outputCommands

   replace FEVTSIMEventContent.outputCommands += toKeepSim1.outputCommands
   replace FEVTSIMEventContent.outputCommands += toKeepSim2.outputCommands
   replace FEVTSIMEventContent.outputCommands += toKeepSim3.outputCommands

   module out = PoolOutputModule {
      using FEVTSIMEventContent
   }
}
""")
            self.assertEqual(t[0].FEVTEventContent.outputCommands,["drop *",
                                                              "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*"])

            self.assertEqual(t[0].FEVTSIMEventContent.outputCommands,["drop *",
                                                                      "drop *",
                                                            "keep blah1_*_*_*",
                                                              "keep blah2_*_*_*",
                                                              "keep blah3_*_*_*",
                                                            "keep blahs1_*_*_*",
                                                              "keep blahs2_*_*_*",
                                                              "keep blahs3_*_*_*"])
            self.assertEqual(t[0].out.outputCommands,
                             t[0].FEVTSIMEventContent.outputCommands)


#NOTE: standard cfg parser can't do the following
            _allUsingLabels = set()
            s="""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block aTest = {
      vstring outputCommands = {"keep blah_*_*_*"}
   }    
   block toKeep = {
      using aTest
   }
   replace outputStuff.outputCommands += toKeep.outputCommands
}
"""
            self.assertRaises(RuntimeError,process.parseString,(s),**dict())
            #self.assertEqual(t[0].outputStuff.outputCommands,["drop *","keep blah_*_*_*"])
            
            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep = {
      vstring outputCommands = {"keep blah_*_*_*"}
   }
   
   block final = {
        using outputStuff
    }
   replace outputStuff.outputCommands += toKeep.outputCommands
}
""")
            self.assertEqual(t[0].outputStuff.outputCommands,["drop *","keep blah_*_*_*"])
            self.assertEqual(t[0].final.outputCommands,["drop *","keep blah_*_*_*"])

            _allUsingLabels = set()
            t=process.parseString("""
process TEST = {
    service = MessageLogger {
        untracked vstring destinations = {"dummy"}
        untracked PSet default = {
               untracked int32 limit = -1
        }
        untracked PSet dummy = {}
    }
    replace MessageLogger.default.limit = 10
    replace MessageLogger.destinations += {"goofy"}
    replace MessageLogger.dummy = { untracked string threshold = "WARNING" }
}""")
            self.assertEqual(t[0].MessageLogger.default.limit.value(),10)
            self.assertEqual(t[0].MessageLogger.destinations,["dummy","goofy"])
            self.assertEqual(t[0].MessageLogger.dummy.threshold.value(),"WARNING")

            _allUsingLabels = set()
            t=process.parseString("""
process TEST = {
  PSet first = {
    int32 foo = 1
    int32 fii = 2
  }
  
  module second = AModule {
    using first
  }
  
  replace first.foo = 2
  
  replace second.fii = 3
}
""")
            self.assertEqual(t[0].first.foo.value(), 2)
            self.assertEqual(t[0].first.fii.value(),2)
            self.assertEqual(t[0].second.foo.value(),2)
            self.assertEqual(t[0].second.fii.value(),3)
            
            _allUsingLabels = set()
            t=process.parseString("""
process TEST = {
    es_module = UnnamedProd {
        int32 foo = 10
    }
    
    es_module me = NamedProd {
        int32 fii = 5
    }
    
    replace UnnamedProd.foo = 1
    
    replace me.fii = 10
}
""")
            self.assertEqual(t[0].UnnamedProd.foo.value(),1)
            self.assertEqual(t[0].me.fii.value(),10)
            
            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
   block outputStuff = {
      vstring outputCommands = {"drop *"}
   }
   block toKeep = {
      vstring outputCommands = {"keep blah_*_*_*"}
   }
   replace outputStuff.outputCommands += toKeep.outputCommands
   
   source = PoolSource {
     untracked vstring fileNames = {"file:foo.root"}
     untracked vint32 foos = {1}
   }
   module out = PoolOutputModule {
     using outputStuff
     untracked string fileName = "blah.root"
   }
   replace PoolSource.fileNames = {"file:bar.root"}
   replace out.fileName = 'blih.root'
   replace PoolSource.foos += 2
   replace PoolSource.foos += 3
}""")
            self.assertEqual(t[0].source.fileNames,["file:bar.root"])
            self.assertEqual(t[0].out.fileName.value(),"blih.root")
            self.assertEqual(t[0].source.foos,[1,2,3])
            self.assertEqual(t[0].outputStuff.outputCommands,["drop *","keep blah_*_*_*"])
            self.assertEqual(t[0].out.outputCommands,["drop *","keep blah_*_*_*"])

            _allUsingLabels = set()
            t=process.parseString("""
process RECO = {
    module foo = FooProd {using b}
    PSet b = {uint32 i = 1}
    PSet c = {using b}
    block p1 = { int32 x = 2
                 int32 y = 9
    }
    PSet p2 = { using p1 }
    module m1 = MProd { using p2 }
    module m2 = MProd { using p2 }
    block b1 = {double y = 1.1 }
    block b2 = {double x = 2.2 }
}
""")
            self.assertEqual(t[0].foo.i.value(),1)
            self.assertEqual(t[0].c.i.value(),1)
            self.assert_(not hasattr(t[0].foo,'using_b') )
            self.assertEqual(t[0].p2.x.value(),2)
            self.assertEqual(t[0].m1.x.value(),2)
            self.assertEqual(t[0].m2.x.value(),2)
            #print t[0].dumpConfig()

            _allUsingLabels = set()
            s="""
process RECO = {
    module foo = FooProd {using b}
    PSet b = {using c}
    PSet c = {using b}
}
"""
            self.assertRaises(RuntimeError,process.parseString,(s),**dict())

            _allUsingLabels = set()
            s="""
process RECO = {
    module foo = FooProd {using b
        uint32 alreadyHere = 1
    }
    PSet b = {uint32 alreadyHere = 2}
}
"""
            self.assertRaises(RuntimeError,process.parseString,(s),**dict())
            #this was failing because of order in which the using was applied

            _allUsingLabels = set()
            t=process.parseString("""
process USER = 
{
        source = PoolInput
        {
                string filename = "here"
        }

        block p1 = {
                    int32 x = 2
                    int32 y = 9
                   }
        PSet  p2 = {
                    using p1
                   }

        module m1 = MidpointJetProducer
        {
                using p2
        }

        module m2 = MidpointJetProducer
        {
                using p2
        }

        block b1 = { double y = 1.1 }
        block b2 = { double x = 2.2 }

        block G = {
          int32 g = 0
        }

        block H = {
          using G
        }

        block I = {
          using J
        }

        block J = {
          int32 j = 0
        }

        replace J.j = 1

        module A = UnconfigurableModule { }
        module B = UnconfigurableModule { }
        module C = UnconfigurableModule { }
        module D = UnconfigurableModule { }
        module E = UnconfigurableModule { }
        module F = UnconfigurableModule { }
        module filter = UnconfigurableFilter { }

        sequence s0a = { A}
        sequence s0b = { A,B}
        sequence s0c = { A&B}
        sequence s1 = { A,B&C,(D,E)&F }
        sequence s2 = { C&(A,B), m1,m2,s1 }
        sequence s3 = {s0a}

        path t1 = { (A,B&C,D),s0a,filter }
        path t2 = { A,B,C,D }
        path t3 = {s3&F}
        endpath te = { A&B }
        
        schedule = {t1,t2}

}
""")
            self.assertEqual(t[0].p2.x.value(), 2)
            self.assertEqual(t[0].m1.x.value(),2)
            self.assertEqual(t[0].m2.x.value(),2)
            self.assertEqual(t[0].J.j.value(),1)
            self.assertEqual(t[0].I.j.value(),1)
            #make sure dump succeeds
            t[0].dumpConfig()

            _allUsingLabels = set()
            t=process.parseString("""
process USER = 
{
    block a = {int32 i = 1}
    PSet b = {PSet c = {using a
       VPSet e={{using a} } } }
    VPSet d = {{using a}, {}}
}""")
            self.assertEqual(t[0].b.c.i.value(),1)
            self.assertEqual(t[0].d[0].i.value(),1)
            self.assertEqual(t[0].b.c.e[0].i.value(), 1)
            #make sure dump succeeds
            t[0].dumpConfig()

            _allUsingLabels = set()
            t=process.parseString("""
process USER = 
{
    block a = {int32 i = 1}
    PSet b = { PSet c = {}
               VPSet g = {} }
    replace b.c = {using a
       VPSet e={{using a} } }
    VPSet d = {{using a}, {}}
}""")
            self.assertEqual(t[0].b.c.i.value(),1)
            self.assertEqual(t[0].d[0].i.value(),1)
            self.assertEqual(t[0].b.c.e[0].i.value(), 1)
            #make sure dump succeeds
            t[0].dumpConfig()

            _allUsingLabels = set()
            t=process.parseString("""
process USER = 
{
    block a = {int32 i = 1}
    PSet b = { PSet c = {} }
    replace b.c = { PSet d = { using a }
       VPSet e={{using a} } }
}""")
            self.assertEqual(t[0].b.c.d.i.value(),1)
            self.assertEqual(t[0].b.c.e[0].i.value(), 1)
            #make sure dump succeeds
            t[0].dumpConfig()

            t=process.parseString("""
process USER = 
{
    block a = {int32 i = 1}
    module b = BWorker { PSet c = {} }
    replace b.c = { PSet d = { using a }
       VPSet e={{using a} } }
}""")
            self.assertEqual(t[0].b.c.d.i.value(),1)
            self.assertEqual(t[0].b.c.e[0].i.value(), 1)
            #make sure dump succeeds
            t[0].dumpConfig()

            t=process.parseString("""
process p = {

  block roster = {
    string thirdBase = 'Some dude'
    PSet outfielders = {
      string rightField = 'Some dude'
    }
  }

  module bums = Ballclub {
    using roster
  }


  module allstars = Ballclub {
    using roster
  }

  replace allstars.thirdBase = 'Chuck Norris'
  replace allstars.outfielders.rightField = 'Babe Ruth'
}""")
            self.assertEqual(t[0].bums.thirdBase.value(), 'Some dude')
            self.assertEqual(t[0].bums.outfielders.rightField.value(), 'Some dude')
            self.assertEqual(t[0].allstars.thirdBase.value(), 'Chuck Norris')
            self.assertEqual(t[0].allstars.outfielders.rightField.value(), 'Babe Ruth')

            _allUsingLabels = set()

            input= """process RECO = {
   es_prefer label = FooESProd {
   }
   es_module label = FooESProd {
   }
}"""

            t=process.parseString(input)
            self.assertEqual(t[0].label.type_(),"FooESProd")
            print t[0].dumpConfig()
            #self.checkRepr(t[0].dumpConfig(), input)
            _allUsingLabels = set()
            t=process.parseString(
"""
process RECO = {
   es_prefer = FooESProd {
   }
   es_module = FooESProd {
   }
}""")
            self.assertEqual(t[0].FooESProd.type_(),"FooESProd")
            print t[0].dumpConfig()

            
        def testPath(self):
            p = cms.Process('Test')
            p.out = cms.OutputModule('PoolOutputModule')
            p.a = cms.EDProducer('AProd')
            p.b = cms.EDProducer('BProd')
            p.c = cms.EDProducer('CProd')
            t=path.parseString('path p = {out}')
            self.assertEqual(str(t[0][1]),'out')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'out')
            #setattr(p,t[0][0],pth)
            #print getattr(p,t[0][0])
            #print pth
#            print t[0][1]
            t=path.parseString('path p = {a,b}')
            self.assertEqual(str(t[0][1]),'(a,b)')            
            self.checkRepr(t[0][1], 'cms.Path(a*b)')
            options = PrintOptions()
            options.isCfg = True
            self.assertEqual(t[0][1].dumpPython(options), 'cms.Path(process.a*process.b)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*b')
            #print pth
#            print t[0][1]
            t=path.parseString('path p = {a&b}')
            self.assertEqual(str(t[0][1]),'(a&b)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+b')
#            print t[0][1]
            t=path.parseString('path p = {a,b,c}')
            self.assertEqual(str(t[0][1]),'((a,b),c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*b*c')
#            print t[0][1]
            t=path.parseString('path p = {a&b&c}')
            self.assertEqual(str(t[0][1]),'((a&b)&c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+b+c')
#            print t[0][1]
            t=path.parseString('path p = {a&b,c}')
            self.assertEqual(str(t[0][1]),'(a&(b,c))')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+b*c')
#            print t[0][1]
            t=path.parseString('path p = {a,b&c}')
            self.assertEqual(str(t[0][1]),'((a,b)&c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*b+c')
#            print t[0][1]
            t=path.parseString('path p = {(a,b)&c}')
            self.assertEqual(str(t[0][1]),'((a,b)&c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*b+c')
#            print t[0][1]
            t=path.parseString('path p = {(a&b),c}')
            self.assertEqual(str(t[0][1]),'((a&b),c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'(a+b)*c')
#            print t[0][1]
            t=path.parseString('path p = {a,(b&c)}')
            self.assertEqual(str(t[0][1]),'(a,(b&c))')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*(b+c)')
#            print t[0][1]
            t=path.parseString('path p = {a&(b,c)}')
            self.assertEqual(str(t[0][1]),'(a&(b,c))')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+b*c')
#            print t[0][1]
            p.d = cms.Sequence(p.a*p.b)
            t=path.parseString('path p = {d,c}')
            self.assertEqual(str(t[0][1]),'(d,c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a*b*c')
#            print t[0][1]
            t=path.parseString('path p = {a&!b}')
            self.assertEqual(str(t[0][1]),'(a&!b)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+~b')
#            print t[0][1]
            t=path.parseString('path p = {!a&!b&!c}')
            self.assertEqual(str(t[0][1]),'((!a&!b)&!c)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'~a+~b+~c')

            t=path.parseString('path p = {a&-b}')
            self.assertEqual(str(t[0][1]),'(a&-b)')
            pth = t[0][1].make(p)
            self.assertEqual(str(pth),'a+cms.ignore(b)')

        @staticmethod
        def strip(value):
            """strip out whitespace & newlines"""
            return  value.replace(' ','').replace('\n','')
        def checkRepr(self, obj, expected):
            # strip out whitespace & newlines
            observe = self.strip(repr(obj))
            expect = self.strip(expected)    
            self.assertEqual(observe, expect)
        def testReplace(self):
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.uint32(2))
            t=replace.parseString('replace a.b = 1')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'1')
            self.checkRepr(t[0][1], "a.b = 1")
            t[0][1].do(process)
            self.assertEqual(process.a.b.value(),1)
            #print t
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.PSet(c=cms.double(2)))
            t=replace.parseString('replace a.b.c = 1')
            self.assertEqual(t[0][0],'a.b.c')
            self.assertEqual(t[0][1].path, ['a','b','c'])
            self.assertEqual(t[0][1].value,'1')
            self.checkRepr(t[0][1], "a.b.c = 1")
            self.assertEqual(type(process.a.b),cms.PSet)
            t[0][1].do(process)
            self.assertEqual(type(process.a.b),cms.PSet)
            self.assertEqual(process.a.b.c.value(),1.0)
            #print t
            t=replace.parseString('replace a.b.c = 1.359')
            self.assertEqual(t[0][0],'a.b.c')
            self.assertEqual(t[0][1].path, ['a','b','c'])
            self.assertEqual(t[0][1].value,'1.359')
            self.checkRepr(t[0][1], "a.b.c = 1.359")
            t[0][1].do(process)
            self.assertEqual(type(process.a.b),cms.PSet)
            self.assertEqual(process.a.b.c.value(),1.359)
            #print t
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.untracked.PSet())
            self.assertEqual(process.a.b.isTracked(),False)
            t=replace.parseString('replace a.b = {untracked string threshold ="DEBUG"}')
            t[0][1].do(process)
            self.assertEqual(type(process.a.b),cms.PSet)
            self.assertEqual(process.a.b.isTracked(),False)
            self.assertEqual(process.a.b.threshold.value(),"DEBUG")
            self.assertEqual(process.a.b.threshold.isTracked(),False)
            #print t
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.PSet(c=cms.untracked.double(2)))
            self.assertEqual(process.a.b.c.value(),2.0)
            self.assertEqual(process.a.b.c.isTracked(),False)
            t=replace.parseString('replace a.b.c = 1')
            self.checkRepr(t[0][1], "a.b.c = 1")
            t[0][1].do(process)
            self.assertEqual(type(process.a.b),cms.PSet)
            self.assertEqual(process.a.b.c.value(),1.0)
            self.assertEqual(process.a.b.c.isTracked(),False)

            t=replace.parseString('replace a.b = "all that"')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'all that')
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.string('thing'))
            self.checkRepr(t[0][1], "a.b = \'all that\'")
            t[0][1].do(process)
            self.assertEqual(process.a.b.value(),'all that')            #print t

            t=replace.parseString('replace a.b = {}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,[])
            self.checkRepr(t[0][1], "a.b = []")

            #print t
            t=replace.parseString('replace a.b = {1, 3, 6}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,['1','3','6'])
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vint32())
            self.checkRepr(t[0][1], "a.b = [1, 3, 6]")

            t[0][1].do(process)
            self.assertEqual(len(process.a.b),3)
            #print t
            t=replace.parseString('replace a.b = {"all", "that"}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,['all','that'])
            self.checkRepr(t[0][1], "a.b = [\'all\', \'that\']")
            t=replace.parseString('replace a.b = {"all that"}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,['all that'])

            t=replace.parseString('replace a.b = { int32 i = 1 }')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(type(t[0][1].value),cms.PSet)
            self.assertEqual(t[0][1].value.i.value(), 1) 
            self.checkRepr(t[0][1], "a.b = cms.PSet(i=cms.int32(1))")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.PSet(j=cms.uint32(5)))
            t[0][1].do(process)
            self.assertEqual(process.a.b.i.value(),1)
            self.assertEqual(getattr(process.a.b,'j',None),None)
            #print t
#            print t[0][1].value
            t=replace.parseString('replace a.b = { { int32 i = 1 } }')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(type(t[0][1].value),cms.VPSet)
            self.assertEqual(t[0][1].value[0].i.value(), 1) 
            self.checkRepr(t[0][1], "a.b = cms.VPSet(cms.PSet(i=cms.int32(1)))")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.VPSet((cms.PSet(j=cms.uint32(5)))))
            t[0][1].do(process)
            self.assertEqual(process.a.b[0].i.value(),1)
            
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vuint32(2))
            t=replace.parseString('replace a.b += 1')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'1')
            self.checkRepr(t[0][1], "a.b.append(1)")
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),[2,1])

            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vuint32(2))
            t=replace.parseString('replace a.b += {1,3}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,['1','3'])
            self.checkRepr(t[0][1], "a.b.extend([1,3])")
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),[2,1,3])
            
            t=replace.parseString('replace a.b += "all that"')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'all that')
            self.checkRepr(t[0][1], "a.b.append(\'all that\')")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vstring('thing'))
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),['thing','all that'])

            t=replace.parseString('replace a.b += {"all that","and more"}')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,['all that','and more'])
            self.checkRepr(t[0][1], "a.b.extend([\'all that\', \'and more\'])")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vstring('thing'))
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),['thing','all that','and more'])

            t=replace.parseString('replace a.b += { int32 i = 1 }')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(type(t[0][1].value),cms.PSet)
            self.assertEqual(t[0][1].value.i.value(), 1) 
            self.checkRepr(t[0][1], "a.b.append(cms.PSet(i=cms.int32(1)))")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd',
                                       b=cms.VPSet((cms.PSet(j=cms.uint32(5)))))
            t[0][1].do(process)
            self.assertEqual(len(process.a.b),2)
            self.assertEqual(process.a.b[1].i.value(),1)
            
            t=replace.parseString('replace a.b += { { int32 i = 1 } }')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(type(t[0][1].value),cms.VPSet)
            self.assertEqual(t[0][1].value[0].i.value(), 1) 
            self.checkRepr(t[0][1], "a.b.extend(cms.VPSet(cms.PSet(i=cms.int32(1))))")
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.VPSet((cms.PSet(j=cms.uint32(5)))))
            t[0][1].do(process)
            self.assertEqual(len(process.a.b),2)
            self.assertEqual(process.a.b[1].i.value(),1)

            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vuint32(2), c=cms.vuint32(1))
            t=replace.parseString('replace a.b += a.c')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'a.c')
            self.checkRepr(t[0][1], "a.b.append(process.a.c)")
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),[2,1])
            
            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.vuint32(2), c=cms.uint32(1))
            t=replace.parseString('replace a.b += a.c')
            self.assertEqual(t[0][0],'a.b')
            self.assertEqual(t[0][1].path, ['a','b'])
            self.assertEqual(t[0][1].value,'a.c')
            self.checkRepr(t[0][1], "a.b.append(process.a.c)")
            t[0][1].do(process)
            self.assertEqual(list(process.a.b),[2,1])

            process = cms.Process("Test")
            foobar = 'cms.InputTag("foobar","","")'
            fooRepr = 'foobar:'
            process.a = cms.EDProducer('FooProd', b=cms.InputTag("bar",""))
            t = replace.parseString('replace a.b = foobar:')
            self.checkRepr(t[0][1], "a.b ='foobar:'")
            t[0][1].do(process)
            self.assertEqual(process.a.b.configValue(),'foobar')                        

            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.VInputTag((cms.InputTag("bar"))))
            t = replace.parseString('replace a.b = {foobar:}')
            self.checkRepr(t[0][1], "a.b = ['foobar:']")
            t[0][1].do(process)
            #self.assertEqual(process.a.b.configValue('',''),'{\nfoobar::\n}\n')                        
            self.assertEqual(list(process.a.b),[cms.InputTag('foobar')])                        

            process = cms.Process("Test")
            process.a = cms.EDProducer('FooProd', b=cms.VInputTag((cms.InputTag("bar"))))
            t = replace.parseString('replace a.b += {foobar:}')
            self.checkRepr(t[0][1], "a.b.extend(['foobar:'])")
            t[0][1].do(process)
            #self.assertEqual(process.a.b.configValue('',''),'{\nfoobar::\n}\n')                        
            self.assertEqual(list(process.a.b),[cms.InputTag("bar"),cms.InputTag('foobar')])                        

            process = cms.Process("Test")
            process.a = cms.EDProducer("FooProd", b = cms.uint32(1))
            t = replace.parseString('replace a.c = 2')
            self.assertRaises(pp.ParseBaseException,t[0][1].do,(process))

    unittest.main()
#try:
    #onlyParameters.setDebug()
#     test = onlyParameters.parseString(
#"""bool blah = True
#untracked bool foo = False #test
#int32 cut = 1
#bool fii = True""")
#    print test
#except pp.ParseBaseException,pe:
#    print pe
#    print pe.markInputline()
#test = letterstart.parseString("aaa")
#test = letterstart.parseString("aaa_")

#test = letterstart.parseString("_a")
#print test

