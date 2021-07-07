#!/usr/bin/env python3
from FWCore.ParameterSet.Types import PSet
import FWCore.ParameterSet.Config as cms
class RunType(PSet):
  def __init__(self,types=['pp_run','pp_run_stage1','cosmic_run','cosmic_run_stage1','hi_run','hpu_run']):
    PSet.__init__(self)
    self.__runTypesDict = {}
    t=[(x,types.index(x)) for x in types ]
    for k,v in t:
      self.__runTypesDict[k] = v
      self.__dict__[k] = v
       
    self.__runType = self.__runTypesDict[types[0]]
    self.__runTypeName = types[0]
    
  def getRunType(self):
    return self.__runType
    
  def getRunTypeName(self):
    return self.__runTypeName
    
  def setRunType(self,rt):
    if isinstance(rt,int): 
      if rt not in self.__runTypesDict.values():
        raise TypeError("%d not a valid Run Type" % rt)
      
      self.__runType = rt
      self.__runTypeName = [k for k, v in self.__runTypesDict.items() if v == rt][0]
      return
    
    if isinstance(rt,str):
      if rt not in self.__runTypesDict.keys():
        raise TypeError("%s not a valid Run Type" % rt)
      
      self.__runTypeName = rt
      self.__runType = self.__runTypesDict[rt]
      
  def __str__(self):
    return "RunType='%s':%d of %s" % (self.__runTypeName,
                                      self.__runType,
                                      self.__runTypesDict )
    
  def __repr__(self):
    return "RunType='%s':%d of %s" % (self.__runTypeName,
                                      self.__runType,
                                      self.__runTypesDict )

