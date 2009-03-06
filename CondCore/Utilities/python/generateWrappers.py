#!/usr/bin/env python

from CondCore.Utilities.payloadInspectorTemplate import *
import string, os, sys

# All those below run in CondCore/XyzPlugin directory

def guessPackage() :
    lines = ( line for line in file('src/plugin.cc')
              if (line.find('CondFormats')>0 and line.find('DataRecord')<0)
    )
    f = lines.next()
    f = f[f.find('CondFormats/')+len('CondFormats/'):]
    return f[:f.find('/')]
 
def getClasses() :
    ret = []
    lines = ( line for line in file('src/plugin.cc') if line[0:3]=='REG')
    for line in lines:
        ret.append(line[line.find(',')+1:line.find(')')])
    return ret

# generate the comment in classes.h
def generateClassesHeader(package):
    header = '/* Condtion Objects'
    footer = '\n */\n\n'
    leader = '\n * '
    newch = file('../../CondFormats/'+package+'/src/classes_new.h','w')
    newch.write(header)
    for cl in getClasses() :
        newch.write(leader+cl)
    newch.write(footer)
    for line in file('../../CondFormats/'+package+'/src/classes.h'):
        newch.write(line)
    newch.close()

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
          
