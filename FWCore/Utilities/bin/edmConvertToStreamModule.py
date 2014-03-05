#! /usr/bin/env python

import subprocess
import shutil
import sys

kProducer = 0
kFilter = 1

def find_file_for_module(name):
  try:
    found = subprocess.check_output(["git","grep", "class *"+name+" *: *public .*edm::EDProducer"])
    type = kProducer
  except:
    try:
      found = subprocess.check_output(["git","grep", "class *"+name+" *: *public .*edm::EDFilter"])
      type = kFilter
    except:
      print "ERROR: Unable to find class declaration for "+name
      exit(1)
  lfound = found.split("\n")
  if len(lfound) > 2:
    print "ERROR: found more than one declaration of "+name
    print lfound
    exit(1)
  s = lfound[0]
  return (s[:s.find(":")],type)
  
def checkout_package(fileName):
  c = fileName.split("/")
  print "checking out "+c[0]+"/"+c[1]
  sparce_checkout = ".git/info/sparse-checkout"
  f = open(sparce_checkout,"r")
  linesInSparse = set(f.readlines())
  f.close()
  linesInSparse.add(c[0]+"/"+c[1]+"\n")
  
  f = open(sparce_checkout+"_new","w")
  for l in linesInSparse:
    if l:
      f.write(l)
  f.close()
  shutil.move(sparce_checkout,sparce_checkout+"_old")
  shutil.move(sparce_checkout+"_new",sparce_checkout)
  subprocess.call(["git","read-tree","-mu","HEAD"])

def edit_file(fileName,moduleType,moduleName):
  print " editting "+fileName
  fOld = open(fileName)
  fNew = open(fileName+"_NEW","w")
  
  lookingForChanges = True
  for l in fOld.readlines():
    if lookingForChanges:
      if -1 != l.find("#include"):
        if moduleType == kProducer:
          if -1 != l.find("FWCore/Framework/interface/EDProducer.h"):
            l='#include "FWCore/Framework/interface/stream/EDProducer.h"\n'
        elif moduleType == kFilter:
          if -1 != l.find("FWCore/Framework/interface/EDFilter.h"):
            l = '#include "FWCore/Framework/interface/stream/EDFilter.h"\n'
      elif -1 != l.find("class"):
        if -1 != l.find(moduleName):
          if moduleType == kProducer:
            if -1 != l.find("edm::EDProducer"):
              l = l.replace("edm::EDProducer","edm::stream::EDProducer<>")
              lookingForChanges = False
          elif moduleType == kFilter:
            if -1 != l.find("edm::EDFilter"):
              l=l.replace("edm::EDFilter","edm::stream::EDFilter<>")
    fNew.write(l)
  fNew.close()
  fOld.close()
  shutil.move(fileName,fileName+"_OLD")
  shutil.move(fileName+"_NEW",fileName)

modules = sys.argv[1:]

for m in modules:
  f,t = find_file_for_module(m)
  if f:
    checkout_package(f)
    edit_file(f,t,m)

