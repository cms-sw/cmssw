#!/usr/bin/env python

baseDirs=[]
import os
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

print l

skipIt=['pytest','climate','xrootdpyfs','theanets','hyperopt','thriftpy']
# climate misses plac
# pytest misses py
# theanets misses plac
# xrootdpyfs misses     from fs.errors import DestinationExistsError, DirectoryNotEmptyError, \
#ImportError: cannot import name DestinationExistsError
#hyperopt misses builtins
#thriftpy misses ply
import importlib
for i in l:
    if i in skipIt: continue
    print "Importing",i,".....",
    importlib.import_module(i)
    print "Done"
