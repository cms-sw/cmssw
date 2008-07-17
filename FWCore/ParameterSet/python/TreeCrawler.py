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

import sys, os, inspect, copy
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
        modulefinder.ModuleFinder.__init__(self,*args,**kwargs)
    def import_hook(self, name, caller=None, fromlist=None):
        old_last_caller = self._last_caller
        try:
            self._last_caller = caller
            return modulefinder.ModuleFinder.import_hook(self,name,caller,fromlist)
        finally:
            self._last_caller = old_last_caller
    def import_module(self,partnam,fqname,parent):
        if partnam in ("FWCore","os"):
            r = None
        else:
            r = modulefinder.ModuleFinder.import_module(self,partnam,fqname,parent)
        if r is not None:
            self._depgraph.setdefault(self._last_caller.__name__,{})[r.__name__] = 1
        return r
    def load_module(self, fqname, fp, pathname, (suffix, mode, type)):
        r = modulefinder.ModuleFinder.load_module(self, fqname, fp, pathname, (suffix, mode, type))
        if r is not None:
            self._types[r.__name__] = type
        return r


def transformIntoGraph(depgraph,toplevel):
    packageDict = {}
    # create the top level config
    packageDict[toplevel] = Package(toplevel, top = True) 

    # create package objects
    for key, value in depgraph.iteritems():
        if key.count(".") == 2: packageDict[key] = Package(key)
        for name in value.keys():
            if name.count(".") == 2: packageDict[name] = Package(name)
    # now create dependencies
    for key, value in depgraph.iteritems():
        if key.count(".") == 2 or key == toplevel:
            package = packageDict[key]
            package.dependencies = [packageDict[name] for name in value.keys() if name.count(".") == 2]

    # find and return the top level config
    return packageDict[toplevel]


def getDependenciesFromConfig(filename,toplevelname,path):
    imports = [filename]
    config = open(filename)
    
    for line in config.readlines():
        # look for all load statements
        if line.startswith("process.load"):
            line = line.replace('"', "'") 
            moduleName = line.split("'")[1]
            module = __import__(moduleName,[],[],"*")
            imports.append(inspect.getsourcefile(module))
    config.close()

    # sort our loaded files alphabetically in Subsytem/Package pattern
    imports.sort(key= lambda x: x.split("python/")[-1])

    # hold a dict of the kind: {package : {dependencies}}
    globalDependencyDict = {}

    # create the flat include dictionaries for each load and merge them
    for item in imports:
        modulefinder = mymf(path)
        modulefinder.run_script(item)

        for key, value in modulefinder._depgraph.iteritems():
            if key == "__main__": key = packageNameFromFilename(item)
            globalDependencyDict[key] = value

    # now set the dependencies of our top level config with the load statements
    globalDependencyDict[toplevelname] = {}
    for item in imports:
        name = packageNameFromFilename(item)
        if name != toplevelname:
            globalDependencyDict[toplevelname][name] = 1
    return globalDependencyDict                                                


def getDependenciesFromPythonFile(filename,toplevelname,path):
    modulefinder = mymf(path)
    modulefinder.run_script(filename)
    globalDependencyDict = modulefinder._depgraph
    globalDependencyDict[toplevelname] = globalDependencyDict["__main__"] 
    return globalDependencyDict


def getImportTree(filename,path):
    config = __import__(filename.rstrip(".py"))
    toplevelname = packageNameFromFilename(filename)
    # get dependencies from given file
    if hasattr(config,"process"):
        globalDependencyDict = getDependenciesFromConfig(filename,toplevelname,path)
    else:
        globalDependencyDict = getDependenciesFromPythonFile(filename,toplevelname,path)
    # transform this flat structure in a dependency tree
    dependencyGraph = transformIntoGraph(globalDependencyDict,toplevelname)
    return dependencyGraph                                               
