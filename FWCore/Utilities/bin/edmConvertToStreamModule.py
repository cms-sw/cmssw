#! /usr/bin/env python

import subprocess
import shutil
import sys
import re

kProducer = 0
kFilter = 1

def find_all_module_classes():
  s = set()
  found = subprocess.check_output(["git","grep", "class *[A-Za-z0-9_<>]* *: *public "])
  s.update(found.split("\n"))

  ret = dict()
  for l in s:
    parts = l.split(":")
    if len(parts)>2:
      file = parts[0]
      name = parts[1]
      name = name[name.find("class")+5:].strip()
      ret.setdefault(name,[]).append((":".join(parts[2:]), file))
  
  return ret

def print_lines(lines):
  for l in lines:
    if len(l)>0:
      print "  ",l

def find_file_for_module(name,all_modules):
  if name in all_modules:
    info = all_modules[name]
    if len(info) != 1:
      print "ERROR: more than one declaration for '"+name+"'\n"
      for inherits,file in info:
        print "  ",file, inherits
      return (None,None)
    inherits,file = info[0]
    if -1 != inherits.find("edm::EDProducer"):
      type = kProducer
    elif -1 != inherits.find("edm::EDFilter"):
      type = kFilter
    else:
      print "ERROR: class '"+name+"' does not directly inherit from EDProducer or EDFilter\n  "+inherits
      return (None,None)
    return (file,type)
  print "ERROR: did not find a standard class declaration for '"+name+"'"
  try:
    found = subprocess.check_output(["git","grep", "class *"+name+" *:"])
    print_lines( found.split("\n") )
  except:
    try:
      found = subprocess.check_output(["git","grep", "typedef *.* "+name])
      print_lines( found.split("\n") )
    except:
      pass
  return (None,None)
  
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
  addedInclude = False
  for l in fOld.readlines():
    if lookingForChanges:
      if -1 != l.find("#include"):
        if moduleType == kProducer:
          if -1 != l.find("FWCore/Framework/interface/EDProducer.h"):
            l='#include "FWCore/Framework/interface/stream/EDProducer.h"\n'
            addedInclude = True
        elif moduleType == kFilter:
          if -1 != l.find("FWCore/Framework/interface/EDFilter.h"):
            l = '#include "FWCore/Framework/interface/stream/EDFilter.h"\n'
            addedInclude = True
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
    if -1 != l.find(" beginJob("):
      print " WARNING: beginJob found but not supported by stream"
      print "  ",l
    if -1 != l.find(" endJob("):
      print " WARNING: endJob found but not supported by stream"
      print "  ",l
  if not addedInclude:
    print " WARNING: did not write include into "+fileName
  fNew.close()
  fOld.close()
  shutil.move(fileName,fileName+"_OLD")
  shutil.move(fileName+"_NEW",fileName)

modules = sys.argv[1:]

print "getting info"
all_mods_info= find_all_module_classes()


for m in modules:
  f,t = find_file_for_module(m,all_mods_info)
  if f:
    checkout_package(f)
    edit_file(f,t,m)

