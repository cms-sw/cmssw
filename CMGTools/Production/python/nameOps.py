#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## October 24th 2011

import re

def getSampleName(name):
    """Return the datasets name in the format /Primary/Proc/Tiers/V-/Pat/etc"""
    try:
        name = removeUser(name)
        name = name.lstrip("/").rstrip("/").split("/")
        proc = name[1]
        algos = proc.split("--")
        proc = algos[0]
        end = ""
        if len(algos)>1:
            end = "/"+  "/".join(algos[1:])
        castor = "/"+name[0]+"/"+proc+"/"+name[2]+end
        return castor
    except:
        return None

def getFileOwner(name):
    try:
        parts = name.lstrip("/").split("/")
        user = parts[1].split("---")[1]
        return user
    except:
        return None

def getCMGDB(name):
    try:
        parts = name.lstrip("/").split("/")
        prim = parts[0]
        proc = parts[1]
        tiers = parts[2]
        if len(parts[3:])>0: proc=proc+"--"
        proc = proc + "--".join(parts[3:])
        return "/" + prim + "/" + proc + "/" + tiers
    except:
        return name

def getCMGDBWithUser(name, user):
    try:
        dbs = getCMGDB(name)
        splitDbs= dbs.lstrip("/").split("/")
        splitProc = splitDbs[1].split("--")
        if len(splitProc) > 1:
            splitDbs[1] += "---" +user
        dbs = "/" + "/".join(splitDbs)
        return dbs
    except:
        return None
        
def getParentWithSampleName(name):
    try:
        name = name.lstrip("/").split("/")
        if len(name) > 3:    
            return "/"+"/".join(name[:-1])
        else: return None
    except: 
        return None
    
def getUnknownParentWithCMGDB(name):
    try:
        
        name = removeUser(name)
        name = name.lstrip("/").split("/")
        if len(name) is 3:
            
            proc = name[1].split("--")
            if len(proc)>2:
                proc = proc[:-1]
                proc = "--".join(proc)
                return "/"+name[0]+"/"+proc+"---*/"+name[2]
            elif len(proc) == 2:
                proc = proc[:-1]
                proc = "--".join(proc)
                return "/"+name[0]+"/"+proc+"/"+name[2]
            else: return None
        else: return None
    except:
        return None 

def getKnownParentWithCMGDB(name,user):
    try:
        name = removeUser(name)
        name = name.lstrip("/").split("/")
        if len(name) is 3:
            proc = name[1].split("--")
            if len(proc)>2:
                proc = proc[:-1]
                proc = "--".join(proc)
                return "/"+name[0]+"/"+proc+"---"+user+"/"+name[2]
            elif len(proc) == 2:
                return "/"+name[0]+"/"+proc+"/"+name[2]
            else: return None
        else: return None
    except:
        return None 
            
def removeUser(setName):
    try:
        parts = setName.lstrip("/").rstrip("/").split("/")
        parts[1] = parts[1].split("---")[0]
        name = "/".join(parts)
        return "/" + name
    except:
        return setName

def isCMGDBName(name):
	length = len(name.rstrip().lstrip().rstrip("/").lstrip("/").split("/"))
	if length == 3 and re.search("---",name):
		return True
	else: return False

def isSampleName(name):
	length = len(name.rstrip().lstrip().rstrip("/").lstrip("/").split("/"))
	if length > 3 and not re.search("---",name):
		return True
	else:
		return False
