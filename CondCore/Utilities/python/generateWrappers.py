#!/usr/bin/env python

from CondCore.Utilities.payloadInspectorTemplate import *
import string, os, sys


def walkPlugins(a):
    for dir in os.listdir('.'):
        if(dir.find('Plugins')<0) : continue
        os.chdir(dir)
        a.act()
        os.chdir('../')


# All those below run in CondCore/XyzPlugin directory

def guessPackage() :
    lines = ( line for line in file('src/plugin.cc')
              if (line.find('CondFormats')>0 and line.find('DataRecord')<0)
    )
    _f = lines.next()
    _f = _f[_f.find('CondFormats/')+len('CondFormats/'):]
    return _f[:_f.find('/')]
 
def guessClasses() :
    _ret = []
    lines = ( line for line in file('src/plugin.cc') if line[0:3]=='REG')
    for line in lines:
        _ret.append(line[line.find(',')+1:line.find(')')])
    return _ret

# generate the comment in classes.h
def generateClassesHeader(package):
    _header = '/* Condition Objects'
    _footer = '\n */\n\n#include "CondFormats/Common/interface/PayloadWrapper.h"\n\n'
    _leader = '\n * '

    _classes = guessClasses()
    
    _newch = file('../../CondFormats/'+package+'/src/classes_new.h','w')
    _newch.write(_header)
    for cl in _classes :
        _newch.write(_leader+cl)
    _newch.write(_footer)
    for line in file('../../CondFormats/'+package+'/src/classes.h'):
        _newch.write(line)
    _newch.close()

def getClasses(package) :
    _header = 'Condition Objects'
    _ret = []
    _ch = file('../../CondFormats/'+package+'/src/classes.h')
    if (_ch.readline().find(_header)<0) :
        print 'comment header not found in '+package
        return _ret
    for line in _ch:
        if (line.find('*/')>0) : break
        _ret.append(line[3:].rstrip())
    _ch.close()
    return _ret

wrapperDeclarationHeader = """
// wrapper declarations
namespace {
   struct wrappers {
"""
wrapperDeclarationFooter = """
   };
}
"""

def declareCondWrappers(package):
    _newch = file('../../CondFormats/'+package+'/src/classes_new.h','w')
    for line in file('../../CondFormats/'+package+'/src/classes.h'):
        if (line.find('wrapper declarations')>0) : break
        _newch.write(line)
    _newch.write(wrapperDeclarationHeader)
    _n=0
    for cl in getClasses(package):
        _newch.write('      pool::Ptr<'+cl+' > p'+str(_n)+';\n')
        _newch.write('      cond::DataWrapper<'+cl+' > dw'+str(_n)+';\n')
        _n=_n+1
    _newch.write(wrapperDeclarationFooter)
    _newch.close()

def declareXMLCondWrappers(package):
    _wrapperComment = '<-- wrapper declarations -->\n'
    _newxml = file('../../CondFormats/'+package+'/src/classes_def_new.xml','w')
    for line in file('../../CondFormats/'+package+'/src/classes_def.xml'):
        if (line.find('wrapper declarations')>0) : break
        _newch.write(line)
    _newch.write_(wrapperComment)
    for cl in getClasses(package):
        _newch.write(' <class name="pool::Ptr<'+cl+' >"/>\n')
        _newch.write(' <class name="cond::DataWrapper<'+cl+' >"/>\n')
    _newch.close()




def generateBuildFile(package,classes) :
    f = file('plugins/BuildFile','w')
    f.seek(0,2)
    s = string.Template(buildfileTemplate)
    for classname in classes:
        f.write(s.substitute(_PACKAGE_=package, _CLASS_NAME_=classname).replace('-',''))

def generateWrapper(package,classes) :
    s = string.Template(wrapperTemplate)
    for classname in classes:
        f = file('plugins/'+classname+'PyWrapper.cc','w')
        print "generating file:", f.name
        f.write(s.substitute(_PACKAGE_=package, _CLASS_NAME_=classname, _HEADER_FILE_=classname))
        f.close()
          

def generateDict(package):
    os.system('cd ../../;cvs co CondFormats/'+package)
    declareCondWrappers(package)
    declareXMLCondWrappers(package)

