#!/usr/bin/env python3
from __future__ import print_function
from builtins import range
import os,subprocess,sys,re,time,random
from threading import *
from subprocess import call
#Some Constants
STATE_CREATED,STATE_COMPLETED,STATE_ERROR=list(range(3)) 

### Classes
class BuildThread(Thread):
  
  def __init__(self, parent, queue, weight = 1):
    Thread.__init__(self)
    self.BuildNode = parent
    self.QueueList = queue
    self.Weight = weight
    self.IsComplete = Condition()
    self.Queue = None
  
  def putInServerQueue(self):
    self.QueueList.QueueLock.acquire()
    sSrv=self.QueueList.smallestQueue()
    tSrv=self.QueueList.thinerQueue()
    self.Queue=sSrv
    if sSrv == tSrv:
      self.QueueList[sSrv].append(self)
    elif self.Weight + self.QueueList[sSrv].queueWeight() <= (self.QueueList.Cores/self.QueueList.Jay)*1.5:
      self.QueueList[sSrv].append(self)
    else:
      self.QueueList[tSrv].append(self)
      self.Queue=tSrv
    self.QueueList.QueueLock.release()
  
  def build(self):
    self.QueueList[self.Queue].QueueSem.acquire()
    #call(['eval',"scram runtime -sh",";",'EdmPluginRefresh',self.BuildNode.LibName],shell="/bin/bash"))
    rValue=call(['ssh',self.Queue,'cd ~/CMSSW_3_5_7;scram build -j %d' % self.QueueList.Jay,self.BuildNode.LibName])
    self.QueueList.EdmRefreshLock.acquire()
    call(['ssh',self.Queue,'cd ~/CMSSW_3_5_7;eval `scram runtime -sh`;EdmPluginRefresh %s' % self.BuildNode.LibName])   
    self.QueueList.EdmRefreshLock.release()
    if rValue == 0: 
      self.BuildNode.State=STATE_COMPLETED
    else:
      print("Build failed for %s"  % self.BuildNode.LibName)
      self.BuildNode.State=STATE_ERROR
    self.QueueList[self.Queue].QueueSem.release()
  
  def releaseAllLocks(self):
    #for deps in self.BuildNode.DependsOn:
      #deps.BThread.IsComplete.release()
    self.BuildNode.State=STATE_ERROR
    self.IsComplete.acquire()
    self.IsComplete.notifyAll()
    self.IsComplete.release()
  
  def run(self):
    depsCompleted=False
    while not depsCompleted:
      depsCompleted=True
      for deps in self.BuildNode.DependsOn:
        if deps.State is STATE_ERROR :
          self.releaseAllLocks()
          return -1
        if deps.State is not STATE_COMPLETED :
          depsCompleted=False
          deps.BThread.IsComplete.acquire() 
          deps.BThread.IsComplete.wait()
          #deps.BThread.isAlive() and sys.stdout.write("Wait time exeded %s %s\n" % (deps.LibName,deps.Module))
          deps.BThread.IsComplete.release()
      
    self.putInServerQueue()
    self.build()
    self.IsComplete.acquire()
    self.IsComplete.notifyAll()
    self.IsComplete.release()
    return 0
    
class BuildTreeNodeList (list):
  
  def __init__(self,value=None):
    self.SeenLibs = []
    self.SeenModules = []
    self.SeenSubModules = []
    if value:
      list.__init__(self,value)
    else:
      list.__init__(self) 
  
  def findLib(self,lib,head=None):
    if len(self) == 0:
      return None
    if head == self:
      return None 
    if head == None:
      head = self
    itP=None
    for it in self:
      if lib == it.LibName:
        return it
      else:
        itP=it.AreDependent.findLib(lib,head)
        if itP is not None:
          return itP 
    return itP
    
  def startThreads(self):
    for node in self:
      if not node.BThread.is_alive():
        try: 
          node.BThread.start()
        except:
          pass
      node.AreDependent.startThreads()
  
  def findDep(self,dep):
    return self.findLib(dep.replace("/",""))
  
  def __setitem__(self,index,value):
    if not value.__class__ == BuildTreeNode:
      raise TypeError("Expected BuildTreeNode") 
    self.SeenLibs.append(value.LibName)
    self.SeenModules.append(value.Module)
    self.SeenSubModules.append(value.SubModule)
    list.__setitem__(self,index,value)
  
  def append(self,value):
    if not value.__class__ == BuildTreeNode:
      raise TypeError("Expected BuildTreeNode") 
    value.LibName   not in self.SeenLibs       and self.SeenLibs.append(value.LibName)
    value.Module    not in self.SeenModules    and self.SeenModules.append(value.Module)
    value.SubModule not in self.SeenSubModules and self.SeenSubModules.append(value.SubModule) 
    list.append(self,value)
  
  def __str__(self,topDown=False,direction=False):
    if not topDown:
      return "[\n...%s]" % str("\n...".join([str(it).replace("...","......") for it in self]))
    else:
      if direction:
        return "[\n---%s]" % "\n---".join([ "'%s':'%s'\n------State = %d \n------DependsOn :  %s " % (it.LibName,it.SubModule or it.Module,it.State,it.DependsOn.__str__(True,True).replace("---","------")) for it in self ])
      else:
        return "\n".join([it.__str__(True) for it in self])
    
class BuildTreeNode (object):
  
  def __init__(self,libname="",module="",submodule="",depends=None,areDependent=None,weight=1,srvqueue=None):
    if depends==None:
      self.DependsOn = BuildTreeNodeList()
    else:
      self.DependsOn = depends
    if areDependent==None:
      self.AreDependent = BuildTreeNodeList()
    else: 
      self.AreDependent = areDependent
    self.Module = module
    self.LibName = libname
    self.SubModule = submodule
    self.BThread = srvqueue is not None and BuildThread(self,srvqueue,weight) or None
    self.State = STATE_CREATED
  
  def __setattr__(self,name,value):
    if name is "DependsOn" or  name is "AreDependent":
      if not value.__class__ == BuildTreeNodeList:
        raise TypeError("Expected BuildTreeNodeList")
    elif name is "State" or name is "ModulesToDo":
      if not value.__class__ == int:
        raise TypeError("Expected int")
    elif name is "Module" or name is "SubModule" or name is "LibName":
      if not value.__class__ == str:
        raise TypeError("Expected str")
    object.__setattr__(self,name,value)

  def __str__(self,topDown=False):
    if not topDown:
      return "'%s':'%s'\n...State = %d , Is it Done = %s \n...AreDependent : %s " % (self.LibName,self.SubModule or self.Module,self.State,self.BThread.IsComplete,self.AreDependent) 
    else:
      if len(self.AreDependent)== 0:
        return "'%s':'%s'\n---State = %d \n---DependsOn :  %s " % (self.LibName,self.SubModule or self.Module,self.State,self.DependsOn.__str__(True,True).replace("---","------"))
      else:
        return self.AreDependent.__str__(topDown=True)
####
# Class to manage server build queues
####   
class queueNode():
  def __init__(self,cores = 4,jay = 2):
    self.QueueSem = BoundedSemaphore(value=cores/jay)
    self.RunningThreads = []
    self.ThreadLog = []
  
  def append(self,item):
    self.RunningThreads.append(item)
    t=time.time()
    self.ThreadLog.append([t,item.BuildNode.LibName,"INFO: %s thread %s added for Library %s" % (t,item.name,item.BuildNode.LibName)])    
  
  def pendingThreads(self):
    return len([x for x in self.RunningThreads if x.is_alive()])
    
  def queueWeight(self):
    return sum([x.Weight for x in self.RunningThreads if x.is_alive()])
  
  
     
class queueList(dict):
  def __init__(self,servers = [],cores = 4,jay = 2):
    dict.__init__(self,dict.fromkeys(servers))
    for srv in self.keys():
      self[srv]=queueNode(cores,jay)
    self.Cores = cores
    self.Jay = jay
    self.QueueLock = RLock() 
    self.EdmRefreshLock = RLock()
  

  def smallestQueue(self):
    smallest=self.keys()[0]
    sizeSmallest=self[smallest].pendingThreads()
    for srv in self.keys()[1:]:
      size=self[srv].pendingThreads()
      if size < sizeSmallest:
        smallest = srv
        sizeSmallest = size
    return smallest
      
  def thinerQueue(self):
    thinnest=self.keys()[0]
    weightThinnest=self[thinnest].queueWeight()
    for srv in self.keys()[1:]:
      weight=self[srv].queueWeight()
      if weight < weightThinnest:
        thinnest = srv
        weightThinnest = weight
    return thinnest
  
    
    
    
   

