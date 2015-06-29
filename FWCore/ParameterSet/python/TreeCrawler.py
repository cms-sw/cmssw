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

import sys, os, inspect, copy, struct, dis, imp
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
                              
        if partnam in ("os","unittest"):
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
        look for "process.load(<module>)' commands. Since the Process object might not necassarily
        be called "process", it scans for a call to a "load" method with a single parameter on
        *any* object. If one is found it checks if the parameter is a string that refers to a valid
        python module in the local or global area. If it does, the scanner assumes this was a call
        to a Process object and yields the module name.
        It's not possible to scan first for Process object declarations to get the name of the
        objects since often (e.g. for customisation functions) the object is passed to a function
        in a different file.

        The ModuleFinder.scan_opcodes_25 implementation this is based was taken from
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
        HAVE_ARGUMENT = modulefinder.HAVE_ARGUMENT
        LOAD_ATTR = chr(dis.opname.index('LOAD_ATTR'))
        LOAD_NAME = chr(dis.opname.index('LOAD_NAME'))
        CALL_FUNCTION = chr(dis.opname.index('CALL_FUNCTION'))
        LOAD_LOAD_AND_IMPORT = LOAD_CONST + LOAD_CONST + IMPORT_NAME
        
        try :
            indexOfLoadConst = names.index("load") # This might throw a ValueError
            # These are the opcodes required to access the "load" attribute. This might
            # not even be a function, but I check for that later.
            loadMethodOpcodes = LOAD_ATTR+struct.pack('<H',indexOfLoadConst)
        except ValueError :
            # doesn't look like "load" is used anywhere in this file
            loadMethodOpcodes=None

        while code:
            c = code[0]
            
            # Check to see if this is a call to a "load" method
            if loadMethodOpcodes!=None and len(code)>=9 : # Need at least 9 codes for the full call
                if code[:3]==loadMethodOpcodes :
                    # The attribute "load" is being accessed, need to make sure this is a function call.
                    # I'll look ahead and see if the CALL_FUNCTION code is used - this could be in a different
                    # place depending on the number of arguments, but I'm only interested in methods with a
                    # single argument so I know exactly where CALL_FUNCTION should be.
                    if code[6]==CALL_FUNCTION :
                        # I know this is calling a method called "load" with one argument. I need
                        # to find out what the argument is. Note that I still don't know if this is
                        # on a cms.Process object.
                        indexInTable=unpack('<H',code[4:6])[0]
                        if code[3]==LOAD_CONST :
                            # The argument is a constant, so retrieve that from the table
                            loadMethodArgument=consts[indexInTable]
                            # I know a load method with one argument has been called on *something*, but I don't
                            # know if it was a cms.Process object. All I can do is check to see if the argument is
                            # a string, and if so if it refers to a python file in the user or global areas.
                            try :
                                loadMethodArgument = loadMethodArgument.replace("/",".")
                                # I can only use imp.find_module on submodules (i.e. each bit between a "."), so try
                                # that on each submodule in turn using the previously found filename. Note that I have
                                # to try this twice, because if the first pass traverses into a package in the local
                                # area but the subpackage has not been checked out it will report that the subpackage
                                # doesn't exist, even though it is available in the global area.
                                try :
                                    parentFilename=[self._localarea+"/python"]
                                    for subModule in loadMethodArgument.split(".") :
                                        moduleInfo=imp.find_module( subModule, parentFilename )
                                        parentFilename=[moduleInfo[1]]
                                    # If control got this far without raising an exception, then it must be a valid python module
                                    yield "import", (None, loadMethodArgument)
                                except ImportError :
                                    # Didn't work in the local area, try in the global area.
                                    parentFilename=[self._globalarea+"/python"]
                                    for subModule in loadMethodArgument.split(".") :
                                        moduleInfo=imp.find_module( subModule, parentFilename )
                                        parentFilename=[moduleInfo[1]]
                                    # If control got this far without raising an exception, then it must be a valid python module
                                    yield "import", (None, loadMethodArgument)
                            except Exception as error:
                                # Either there was an import error (not a python module) or there was a string
                                # manipulaton error (argument not a string). Assume this wasn't a call on a
                                # cms.Process object and move on silently.
                                pass
                        
                        elif code[3]==LOAD_NAME :
                            # The argument is a variable. I can get the name of the variable quite easily but
                            # not the value, unless I execute all of the opcodes. Not sure what to do here,
                            # guess I'll just print a warning so that the user knows?
                            print "Unable to determine the value of variable '"+names[indexInTable]+"' to see if it is a proces.load(...) statement in file "+co.co_filename
                        
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

def removeRecursiveLoops( node, verbose=False, currentStack=None ) :
    if currentStack is None : currentStack=[]
    try :
        duplicateIndex=currentStack.index( node ) # If there isn't a recursive loop this will raise a ValueError
        if verbose :
            print "Removing recursive loop in:"
            for index in xrange(duplicateIndex,len(currentStack)) :
                print "   ",currentStack[index].name,"-->"
            print "   ",node.name
        currentStack[-1].dependencies.remove(node)
    except ValueError:
        # No recursive loop found, so continue traversing the tree
        currentStack.append( node )
        for subnode in node.dependencies :
            removeRecursiveLoops( subnode, verbose, currentStack[:] )

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

    removeRecursiveLoops( packageDict[toplevel] )
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
