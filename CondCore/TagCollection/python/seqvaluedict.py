#  Sequential Dictionary Class                                                 #
'''
The sequential dictionary is a combination of a list and a dictionary so you can do most operations defined with lists . 
seqdict - single value dictionary , keeps one value for one key 
'''
from functools import reduce

class seqdict:
  def __init__(self,List=[],Dict={}):
    if type(List)==type({}):
      self.list = List.keys()
      self.dict = List.copy()
    elif List and not Dict:
      self.list=[]
      self.dict={}
      for i,j in List:
        self.list.append(i)
        self.dict[i]=j
    elif type(List)==type(Dict)==type([]):
      self.list = List
      self.dict = {}
      for key,value in map(None,List,Dict):
        self.dict[key] = value
    else:
      self.list,self.dict = List[:],Dict.copy()
      
  def append(self,key,value):
    if key in self.dict:
      self.list.remove(key)
    self.list.append(key)
    self.dict[key]=value
  def check(self):
    if len(self.dict)==len(self.list):
      l1=self.list[:];l1.sort()
      l2=self.dict.keys();l2.sort()
      return l1==l2
    return -1
  def clear(self):
    self.list=[];self.dict={}
  def copy(self):
    if self.__class__ is seqdict:
      return self.__class__(self.list,self.dict)
    import copy
    return copy.copy(self)
  def __cmp__(self,other):
    return cmp(self.dict,other.dict) or cmp(self.list,other.list)
  def __getitem__(self,key):
    if type(key)==type([]):
      newdict={}
      for i in key:
        newdict[i]=self.dict[i]
      return self.__class__(key,newdict)
    return self.dict[key]
  def __setitem__(self,key,value):
    if key not in self.dict:
      self.list.append(key)
    self.dict[key]=value
  def __delitem__(self, key):
    del self.dict[key]
    self.list.remove(key)      
  def __getslice__(self,start,stop):
    start = max(start,0); stop = max(stop,0)
    newdict = self.__class__()
    for key in self.list[start:stop]:
      newdict.dict[key]=self.dict[key]
    newdict.list[:]=self.list[start:stop]
    return newdict
  def __setslice__(self,start,stop,newdict):
    start = max(start,0); stop = max(stop,0)
    delindexes = []
    for key in newdict.keys():
      if key in self.dict:
        index = self.list.index(key)
        delindexes.append(index)
        if index < start:
          start = start - 1
          stop  = stop  - 1
        elif index >= stop:
          pass
        else:
          stop  = stop - 1
    delindexes.sort()
    delindexes.reverse()
    for index in delindexes:
      key = self.list[index]
      del   self.dict[key]
      del  self.list[index]
    for key in self.list[start:stop]:
      del self.dict[key]
    self.list[start:stop] = newdict.list[:]
    self.update(newdict.dict)
  def __delslice__(self, start, stop):
      start = max(start, 0); stop = max(stop, 0)
      for key in self.list[start:stop]:
        del self.dict[key]
      del self.list[start:stop]
  def __add__(self,other):
    newdict = self.__class__()
    for key,value in self.items()+other.items():
      newdict.append(key,value)
    return newdict    
  def __radd__(self,other):
    newdict = self.__class__()
    for key,value in other.items()+self.items():
      newdict.append(key,value)
    return newdict   
  def count(self,value):
    vallist = self.dict.values()
    return vallist.count(value)
  def extend(self,other):
    self.update(other)
  def filter(self,function):
    liste=filter(function,self.list)
    dict = {}
    for i in liste:
      dict[i]=self.dict[i]
    return self.__class__(liste,dict)
  def get(self, key, failobj=None):
        return self.dict.get(key, failobj)
  def index(self,key):return self.list.index(key)
  def insert(self,i,x):self.__setslice__(i,i,x)
  def items(self):return map(None,self.list,self.values())
  def has_key(self,key):return key in self.dict
  def keys(self):return self.list
  def map(self,function):
    return self.__class__(map(function,self.items()))
  def values(self):
    nlist = []
    for key in self.list:
      nlist.append(self.dict[key])
    return nlist
  def __len__(self):return len(self.list)
  def pop(self,key=None):
    if key==None:
      pos = -1
      key = self.list[pos]
    else:
      pos = self.list.index(key)
    tmp = self.dict[key]
    del self.dict[key]
    return {self.list.pop(pos):tmp}
  def push(self,key,value):
    self.append(key,value)
  def reduce(self,function,start=None):
    return reduce(function,self.items(),start)
  def remove(self,key):
    del self.dict[key]
    self.list.remove(key)
  def reverse(self):self.list.reverse()
  def sort(self,*args):self.list.sort(*args)
  def split(self,function,Ignore=None):
    splitdict = seqdict() #self.__class__()
    for key in self.list:
      skey = function(key)
      if skey != Ignore:
        if skey not in splitdict:
          splitdict[skey] = self.__class__()
        splitdict[skey][key] = self.dict[key]
    return splitdict
  def swap(self):
    tmp = self.__class__(map(lambda x_y:(x_y[1],x_y[0]),self.items()))
    self.list,self.dict = tmp.list,tmp.dict
  def update(self,newdict):
    for key,value in newdict.items():
      self.__setitem__(key,value)
  def slice(self,From,To=None,Step=1):
    From       = self.list.index(From)
    if To:To   = self.list.index(To)
    else :
      To = From + 1
    List = range(From,To,Step)
    def getitem(pos,self=self):return self.list[pos]
    return self.__getitem__(map(getitem,List))
  def __repr__(self):return 'seqdict(\n%s,\n%s)'%(self.list,self.dict)

