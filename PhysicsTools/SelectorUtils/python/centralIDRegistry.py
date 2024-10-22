import FWCore.ParameterSet.Config as cms

class CentralIDRegistry:
   def __init__(self):
       self.md5toName = {}
       self.nameToMD5 = {}

   def register(self,name,md5):
       #register md5 -> name
       if md5 not in self.md5toName:
           self.md5toName[md5] = name
       else:
           raise Exception('md5 %s already exists with name %s!'%(md5,self.md5toName[md5]))
       # register name -> md5
       if name.value() not in self.nameToMD5:
           self.nameToMD5[name.value()] = md5
       else:
           raise Exception('Name %s already exists with md5 %s!'%(name.value(),self.nameToMD5[name.value()]))

   def getNameFromMD5(self,md5):
       if md5 in self.md5toName:
           return self.md5toName[md5]
       else:
           return ''

   def getMD5FromName(self,name):
       if name.value() in self.nameToMD5:
           return self.nameToMD5[name.value()]
       else:
           return ''


central_id_registry = CentralIDRegistry()

   
           
       
