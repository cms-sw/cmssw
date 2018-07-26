#!/usr/bin/env python
from __future__ import print_function
import traceback
import os, sys
#import gc

#class Importer:
#    def __init__(self,name):
#        self.error=0
#        try:
#            importlib.import_module(i)
#            print "Done"
#        except Exception as e:
#            print "Fail"
#            print traceback.format_exc()
#            self.error = 1
#        
#    def err(self):
#        return self.error

baseDirs=[]
from subprocess import Popen,PIPE
for p in ['py2-pippkgs','py2-pippkgs_depscipy']:
    comm="scram tool info "+p+" | grep PYTHONPATH | cut -f2 -d="
    
    p=Popen(comm,stdout=PIPE,shell=True)
    baseDir=p.stdout.read().strip()
    baseDirs.append(baseDir)
    
l=[]
for baseDir in baseDirs:
    for root, dirs, files in os.walk(baseDir):
        for file in files:
            if file.endswith('.py'):
                l.append(file[:-3])
        for file in dirs:
            if 'egg-info' in file: continue
            if 'dist-info' in file: continue
            l.append(file)
        break

#print l
skipIt=['pytest','climate','theanets','hyperopt','thriftpy']
# climate misses plac
# pytest misses py
# theanets misses plac
#hyperopt misses builtins
#thriftpy misses ply
#unless the llvm version in cmssw matches that of root, uproot needs to be
#tested separately (https://github.com/scikit-hep/uproot/issues/58)


#import importlib
#err = 0
for i in l:
    if i in skipIt: continue
    print(i)
#
#    if i in skipIt: continue
#    print "Importing",i,".....",
#    d=Importer(i)
#    err+=d.err()
#    del d
#    gc.collect()
#
#sys.exit(err)
