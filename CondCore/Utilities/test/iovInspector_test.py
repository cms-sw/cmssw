# test of iovInpector

import os,os.path,string,commands
import pluginCondDBPyInterface as CondDB
from CondCore.Utilities import iovInspector as inspect


class How(int) :
   def  __init__(self):
        self.a=0
        self.b=1
        self.c=2


class TestWhat :
    """
    a fake What class
    """
    def __init__(self) :
        self.__how = How()
        self.__which = CondDB.VInt()
        self.__lost = CondDB.VInt()
    def how(self) :
        return self.__how
    def which(self):
        return self.__which
    def lost(self):
        return self.__lost
    def set_how(self,h):
        self.__how=h
    def set_which(self,vi) :
        self.__which = vi
    def set_lost(self,vi) :
        self.__lost= vi
    def descr_which(self):
        d = CondDB.VString()
        d.append('me')
        d.append('you')
        d.append('her')
        return d;

ts = TestWhat()
wd = inspect.WhatDescription(ts)
d = wd.describe()
print d

ret={}
for key in d.keys() :
    (mode,val) = d[key]
    if (mode=='single') :
        ret[key]=val[1]
    else :
        ret[key]=[0,2]

inspect.setWhat(ts,ret)

print ts.how()
print [v for v in ts.which()]
print [v for v in ts.lost()]
