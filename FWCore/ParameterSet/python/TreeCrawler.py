# This CMS code is based on previous work done by Toby Dickenson, as indiciated below
#
# for questions: Benedikt.Hegner@cern.ch

# Copyright 2004 Toby Dickenson
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys, os, inspect, copy, struct, dis
import modulefinder

def packageNameFromFilename(name):
    return ".".join(name.replace("python/","").replace(".py","").split("/")[-3:])


class Color:
  """ANSI escape display sequences"""
  info          = "\033[1;34m"
  hilight       = "\033[31m"
  alternate     = "\033[32m"
  extra         = "\033[33m"
  backlight     = "\033[43m"
  underline     = "\033[4m"
  lessemphasis  = "\033[30m"
  deemphasis    = "\033[1;30m"
  none          = "\033[0m"

_stack = []

class SearchHit:
    pass

class Package(object):
    def __init__(self,name,top=False):
        self.name = name
        self.dependencies = []
        self.searched = False
        self.stack = []
        if top:
            self.module = None
        else:    
            self.module = __import__(name,[],[],"*")
    def dump(self,level):
        indent = "  " * level
        print indent, "+", Color.info, self.name, Color.none
        # sort dependencies alphabetically
        self.dependencies.sort(key = lambda x: x.name)
        for package in self.dependencies:
            package.dump(level+1)
    def search(self,pattern,result):
        """ recursive search for pattern in source files"""
        # first start searching in the package itself / do this only once
        if self.module:
            for number, line in enumerate(inspect.getsource(self.module).splitlines()):
                if pattern in line:
                     filename = packageNameFromFilename(inspect.getsourcefile(self.module))
                     if not self.searched:
                         # save the hit, so we can add later stacks to it
                         self.hit = SearchHit()
                         self.hit.number = number
                         self.hit.filename = filename
                         self.hit.line = line
                         self.hit.stacks = list()
                         result.append(self.hit)
                     self.hit.stacks.append(copy.copy(_stack)) 
        # then go on with dependencies
        _stack.append(self.name)
        for package in self.dependencies:
            package.search(pattern,result)
        _stack.pop() 
        self.searched = True    


class mymf(modulefinder.ModuleFinder):
    def __init__(self,*args,**kwargs):
        self._depgraph = {}
        self._types = {}
        self._last_caller = None
        #TODO - replace by environment variables CMSSW_BASE and CMSSW_RELEASE_BASE (*and* do it only if the global one is not empty like for IB areas)  
        self._localarea = os.path.expandvars('$CMSSW_BASE')
        self._globalarea = os.path.expandvars('$CMSSW_RELEASE_BASE')
        modulefinder.ModuleFinder.__init__(self,*args,**kwargs)
    def import_hook(self, name, caller=None, fromlist=None, level=-1):
        old_last_caller = self._last_caller
        try:
            self._last_caller = caller
            return modulefinder.ModuleFinder.import_hook(self,name,caller,fromlist, level=level)  
        finally:
            self._last_caller = old_last_caller

    def import_module(self,partnam,fqname,parent):
                              
        if partnam in ("FWCore","os","unittest"):
            r = None
        else:
            r = modulefinder.ModuleFinder.import_module(self,partnam,fqname,parent)
            # since the modulefinder is not able to look into the global area when coming from the local area, we force a second try   
            if parent and not r and self._localarea != '' and self._globalarea != '':
                 parent.__file__ = parent.__file__.replace(self._localarea,self._globalarea)
                 parent.__path__[0] = parent.__path__[0].replace(self._localarea,self._globalarea)
            r = modulefinder.ModuleFinder.import_module(self,partnam,fqname,parent)
                                                         
        if r is not None:
            self._depgraph.setdefault(self._last_caller.__name__,{})[r.__name__] = 1
        return r
    def load_module(self, fqname, fp, pathname, (suffix, mode, type)):
        r = modulefinder.ModuleFinder.load_module(self, fqname, fp, pathname, (suffix, mode, type))
        if r is not None:
            self._types[r.__name__] = type
        return r

    def scan_opcodes_25(self, co, unpack = struct.unpack):
        """
        This is basically just the default opcode scanner from ModuleFinder, but extended to also
        look for "process.load(<module>)' commands. This is complicated by the fact that we don't
        know what the name of the Process object is (usually "process", but doesn't have to be).
        So we have to also scan for declarations of Process objects. This is in turn is complicated
        by the fact that we don't know how FWCore.ParameterSet.Config has been imported (usually
        "... as cms" but doesn't have to be) so we also have to scan for that import.
        
        So, the additional parts are:
        
        1) Scan for the FWCore.ParameterSet.Config import and note down what name it's imported as.
        2) Scan for Process declarations using the name noted in (1), record any of the object names.
        3) Scan for "load" method calls to anything noted in (2) and yield their arguments.
        
        The ModuleFinder.scan_opcodes_25 implementation I based this on I got from
        https://hg.python.org/cpython/file/2.7/Lib/modulefinder.py#l364
        """
        # Scan the code, and yield 'interesting' opcode combinations
        # Python 2.5 version (has absolute and relative imports)
        code = co.co_code
        names = co.co_names
        consts = co.co_consts
        LOAD_CONST = modulefinder.LOAD_CONST
        IMPORT_NAME = modulefinder.IMPORT_NAME
        STORE_OPS = modulefinder.STORE_OPS
        STORE_NAME = modulefinder.STORE_NAME
        HAVE_ARGUMENT = modulefinder.HAVE_ARGUMENT
        LOAD_NAME = chr(dis.opname.index('LOAD_NAME'))
        LOAD_ATTR = chr(dis.opname.index('LOAD_ATTR'))
        LOAD_LOAD_AND_IMPORT = LOAD_CONST + LOAD_CONST + IMPORT_NAME
        
        try :
            fwCoreIndex=names.index('FWCore.ParameterSet.Config')
            loadMethodOpargs=struct.pack( '<H', names.index('load') )
            loadMethodOpcodes=(LOAD_ATTR,loadMethodOpargs[0],loadMethodOpargs[1])
            # This will be the list of Process objects. Usually there's just one called "process", but
            # I don't know that for sure. For now I'll just set it to None though, as a marker that I
            # don't yet know the opcodes to declare a process object. Once I do I'll set to "[]"
            processObjects=None
            # This will be the opcodes equivalent to a "foo = cms.Process("bar")" statement. This list as
            # it stands is incomplete, I can't do the rest until I know how FWCore is imported. Whatever
            # happens I need to add [LOAD_NAME,indexOfName,indexOfName] to the start, but I don't know
            # what the name is (usually "cms" because it's imported with "import ... as cms"). I also
            # might need to trim off these LOAD_ATTR commands if they're done before storing the name.
            processDefinitionOpcodes=[]
            for loadAttrName in ['ParameterSet','Config','Process'] :
                opArgument=struct.pack( '<H', names.index(loadAttrName) )
                processDefinitionOpcodes.extend( [LOAD_ATTR,opArgument[0],opArgument[1]] )
        except ValueError : # Doesn't look like this file imports FWCore.ParameterSet.Config and/or use Process
            fwCoreIndex=None
            processObjects=None

        while code:
            c = code[0]
            
            # If I have a full list of the opcodes needed for a Process definition, check to see if this is one
            if processObjects!=None : # processObjects will be None until I have completed step (1) in the comment at the top
                if len(code)>=len(processDefinitionOpcodes) : # Check for step (2) in comment at top
                    isProcessDefinition=True
                    for index in xrange( len(processDefinitionOpcodes) ) :
                        if code[index]!=processDefinitionOpcodes[index] :
                            isProcessDefinition=False
                    if isProcessDefinition :
                        code=code[len(processDefinitionOpcodes):] # Trim off what I've just checked
                        # The only thing I'm interested in is what name is given to the new object,
                        # so that I can search for "<newname>.load( <module> )".
                        while code[0]!=STORE_NAME :
                            if code[0] >= HAVE_ARGUMENT:
                                code = code[3:]
                            else:
                                code = code[1:]
                        # I've hit the opcode which tells me what the name of the object is (probably "process"
                        # but I don't know for sure). Note that I'm storing the opcodes for retrieving the object
                        # rather than a string of the object name 
                        processObjects.append( (LOAD_NAME,code[1],code[2]) )
                        continue
    
                # Wasn't a declaration of a new Process object. See if it is accessing a pre existing one
                if len(code)>=9 : # Check for step (3) in comment at top
                    for processObject in processObjects :
                        if processObject==(code[0],code[1],code[2]) :
                            # One of the process objects is being accessed. See if it's calling the "load" method
                            if (code[3],code[4],code[5])==loadMethodOpcodes :
                                if code[6]==LOAD_CONST :
                                    moduleNameIndex=unpack('<H',code[7:9])[0]
                                    yield "import", (None, consts[moduleNameIndex])
                                    code=code[9:]
                                    continue

            if c in STORE_OPS:
                oparg, = unpack('<H', code[1:3])
                yield "store", (names[oparg],)
                code = code[3:]
                continue
            if code[:9:3] == LOAD_LOAD_AND_IMPORT:
                oparg_1, oparg_2, oparg_3 = unpack('<xHxHxH', code[:9])
                level = consts[oparg_1]

                # See if this is the import of FWCore.ParameterSet.Config (i.e. step (1) in the comment at the top).
                # If processObjects is not None then I've already done this once and don't need to do it again.
                if fwCoreIndex!=None and processObjects==None:
                    if oparg_3==fwCoreIndex :
                        # Peek ahead in the opcodes to see how FWCore.ParameterSet.Config is
                        # stored. Note that I want to peek ahead, so I can't change "code".
                        opcodes=code[9:]
                        while opcodes[0]!=STORE_NAME : # STORE_NAME is the final thing I'm looking for, but I also want to know about LOAD_ATTR
                            if (opcodes[0],opcodes[1],opcodes[2])==(processDefinitionOpcodes[0],processDefinitionOpcodes[1],processDefinitionOpcodes[2]) :
                                # The LOAD_ATTR is done before the save, so I don't need to do it after the load.
                                processDefinitionOpcodes=processDefinitionOpcodes[3:]
                            opcodes=opcodes[3:]
                        # The next command in opcodes is the STORE_NAME. I need to know the arguments because they're
                        # the same as will be used in LOAD_NAME, which goes at the start of the commands.
                        processDefinitionOpcodes=[LOAD_NAME,opcodes[1],opcodes[2]]+processDefinitionOpcodes
                        # I now have the full list of opcodes of when a cms.Process object is declared. I can search for
                        # Process object definitions, each of which I'll put in processObjects.
                        processObjects=[]
                    # I still want to report the import, so don't do a "continue" here

                if level == -1: # normal import
                    yield "import", (consts[oparg_2], names[oparg_3])
                elif level == 0: # absolute import
                    yield "absolute_import", (consts[oparg_2], names[oparg_3])
                else: # relative import
                    yield "relative_import", (level, consts[oparg_2], names[oparg_3])
                code = code[9:]
                continue
            if c >= HAVE_ARGUMENT:
                code = code[3:]
            else:
                code = code[1:]


def transformIntoGraph(depgraph,toplevel):
    packageDict = {}
    # create the top level config
    packageDict[toplevel] = Package(toplevel, top = True) 

    # create package objects
    for key, value in depgraph.iteritems():
        if key.count(".") == 2 and key != toplevel: 
            packageDict[key] = Package(key)
        for name in value.keys():
            if name.count(".") == 2: packageDict[name] = Package(name)
    # now create dependencies
    for key, value in depgraph.iteritems():
        if key.count(".") == 2 or key == toplevel:
            package = packageDict[key]
            package.dependencies = [packageDict[name] for name in value.keys() if name.count(".") == 2]

    # find and return the top level config
    return packageDict[toplevel]


def getDependenciesFromPythonFile(filename,toplevelname,path):
    modulefinder = mymf(path)
    modulefinder.run_script(filename)
    globalDependencyDict = modulefinder._depgraph
    globalDependencyDict[toplevelname] = globalDependencyDict["__main__"] 
    return globalDependencyDict


def getImportTree(filename,path):
    toplevelname = packageNameFromFilename(filename)
    # get dependencies from given file
    globalDependencyDict = getDependenciesFromPythonFile(filename,toplevelname,path)
        
    # transform this flat structure in a dependency tree
    dependencyGraph = transformIntoGraph(globalDependencyDict,toplevelname)
    return dependencyGraph                                               
