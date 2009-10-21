import FWCore.ParameterSet.Config as cms

def new___init__(self,name):
    self.old___init__(name)
    self.__dict__['_Process__history'] = []
    self.__dict__['_Process__enableRecording'] = 0
    self.__dict__['_Process__modifiedmodules'] = []
cms.Process.old___init__=cms.Process.__init__
cms.Process.__init__=new___init__

def new_modifiedModules(self):
    return self.__dict__['_Process__modifiedmodules']
cms.Process.modifiedModules=new_modifiedModules

def new_resetModifiedModules(self):
    self.__dict__['_Process__modifiedmodules'] = []
cms.Process.resetModifiedModules=new_resetModifiedModules

def new__place(self, name, mod, d):
    self.old__place(name, mod, d)
    if self._okToPlace(name, mod, d):
        self.__dict__['_Process__modifiedmodules'].append(mod)
cms.Process.old__place=cms.Process._place
cms.Process._place=new__place

def new__placeSource(self, name, mod):
    self.old__placeSource(name, mod)
    self.__dict__['_Process__modifiedmodules'].append(mod)
cms.Process.old__placeSource=cms.Process._placeSource
cms.Process._placeSource=new__placeSource

def new__placeLooper(self, name, mod):
    self.old__placeLooper(name, mod)
    self.__dict__['_Process__modifiedmodules'].append(mod)
cms.Process.old__placeLooper=cms.Process._placeLooper
cms.Process._placeLooper=new__placeLooper

def new__placeService(self, typeName, mod):
    self.old__placeService(typeName, mod)
    self.__dict__['_Process__modifiedmodules'].append(mod)
cms.Process.old__placeService=cms.Process._placeService
cms.Process._placeService=new__placeService

def new_setSchedule_(self, sch):
    self.old_setSchedule_(sch)
    self.__dict__['_Process__modifiedmodules'].append(sch)
cms.Process.old_setSchedule_=cms.Process.setSchedule_
cms.Process.setSchedule_=new_setSchedule_

def new_setLooper_(self, lpr):
    self.old_setLooper_(lpr)
    self.__dict__['_Process__modifiedmodules'].append(lpr)
cms.Process.old_setLooper_=cms.Process.setLooper_
cms.Process.setLooper_=new_setLooper_

def new_history(self):
    return self.__dict__['_Process__history']
cms.Process.history=new_history
def new_resetHistory(self):
    self.__dict__['_Process__history'] = []
    self.resetModified()
    self.resetModifiedModules()
cms.Process.resetHistory=new_resetHistory

def new_addAction(self,tool):
    if self.__dict__['_Process__enableRecording'] == 0:
        self.__dict__['_Process__history'].append(tool)
cms.Process.addAction=new_addAction

def new_deleteAction(self,i):
    del self.__dict__['_Process__history'][i]
cms.Process.deleteAction=new_deleteAction

def new_disableRecording(self):
    if self.__dict__['_Process__enableRecording'] == 0:
        modification = self.dumpModified()[1]
        if modification!="":
            self.__dict__['_Process__history'].append(modification)
        self.resetModified()
    self.__dict__['_Process__enableRecording'] += 1
cms.Process.disableRecording=new_disableRecording

def new_enableRecording(self):
    self.__dict__['_Process__enableRecording'] -= 1
    if self.__dict__['_Process__enableRecording'] == 0:
        self.resetModified()
cms.Process.enableRecording=new_enableRecording

def new_recurseResetModified_(self, o):
    properties = []
    if hasattr(o, "resetModified"):
        o.resetModified()
    if hasattr(o, "parameterNames_"):
        for key in o.parameterNames_():
            value = getattr(o, key)
            self.recurseResetModified_(value)
cms.Process.recurseResetModified_=new_recurseResetModified_

def new_recurseDumpModified_(self, name, o):
    """
    dumpPython = ""
    if hasattr(o, "parameterNames_"):      
        for key in o.parameterNames_():
            value = getattr(o, key)
            dump = self.recurseDumpModified_(name + "." + key, value)
            if dumpPython != "" and dump != "":
                dumpPython += "\n"
            dumpPython += dump
    elif hasattr(o, "isModified") and o.isModified():
        if isinstance(o, InputTag):
            pythonValue="\"" + str(o.value()) + "\""
        elif hasattr(o, "pythonValue"):
            pythonValue=o.pythonValue()
        elif hasattr(o, "value"):
            pythonValue=o.value()
        else:
            pythonValue=o
        dump = "process." + name + " = " + str(pythonValue)
        if dumpPython != "" and dump != "":
            dumpPython += "\n"
        dumpPython += dump
    return dumpPython
    """
    
    # gfball's interpretation.
    # Recurse over items that are Parameterizable (all modules, parametersets), printing out modifications as comments then the current value as python.
    # This should cover everything defined in items_() below except for sequence types. However, looking at the code although _ModuleSequenceType has a parameter '_isModified' there are no conditions under which it is set.
    # 
    dumpPython = ""
    # Test this is a parameterizable object. This ignores any parameters never placed in a PSet, but I don't think they're interesting anyway?
    if isinstance(o, cms._Parameterizable):
        
        # Build a dictionary parametername->[modifications of that param,...] so that we group all modification statements for a single parameter together.
        mod_dict = {}          
        for mod in o._modifications:
            if mod['name'] in mod_dict:
                mod_dict[mod['name']] += [mod]
            else:
                mod_dict[mod['name']] = [mod]
        
        # Loop over modified parameters at this level, printing them
        for paramname in mod_dict:
            for mod in mod_dict[paramname]:
                dumpPython += "# MODIFIED BY %(file)s:%(line)s; %(old)s -> %(new)s\n" % mod
            dumpPython += "process.%s.%s = %s\n\n" % (name,paramname,getattr(o,paramname)) # Currently, _Parameterizable doesn't check __delattr__ for modifications. We don't either, but if anyone does __delattr__ then this will fail.
            
        # Loop over any child elements
        for key in o.parameterNames_():
            value = getattr(o,key)
            dumpPython += self.recurseDumpModified_("%s.%s"%(name,key),value)
    
    # Test if we have a VPSet (I think the code above would miss checking a VPSet for modified children too)
    if isinstance(o, cms._ValidatingListBase):
        for index,item in enumerate(o):
            dumpPython += self.recurseDumpModified_("%s[%s]"%(name,index),item)
    return dumpPython    
cms.Process.recurseDumpModified_=new_recurseDumpModified_



def new_resetModified(self):
    modification = self.dumpModified()[0]
    self.__dict__['_Process__modifiedmodules'].extend(modification)
    for name, o in self.items_():
        self.recurseResetModified_(o)
cms.Process.resetModified=new_resetModified

def new_dumpModified(self):
    dumpModified = ""
    modifiedObjects = []
    for name, o in self.items_():
        dumpPython = self.recurseDumpModified_(name, o)
        if dumpPython != "":
            if dumpModified != "":
                dumpModified += "\n"
            dumpModified += dumpPython
            if not o in modifiedObjects:
                modifiedObjects += [o]
    return (modifiedObjects, dumpModified)
cms.Process.dumpModified=new_dumpModified

def new_moduleItems_(self):
    items = []
    items += self.producers.items()
    items += self.filters.items()
    items += self.analyzers.items()
    return tuple(items)
cms.Process.moduleItems_=new_moduleItems_

def new_items_(self):
    items = []
    if self.source:
        items += [("source", self.source)]
    if self.looper:
        items += [("looper", self.looper)]
    items += self.moduleItems_()
    items += self.outputModules.items()
    items += self.sequences.items()
    items += self.paths.iteritems()
    items += self.endpaths.items()
    items += self.services.items()
    items += self.es_producers.items()
    items += self.es_sources.items()
    items += self.es_prefers.items()
    items += self.psets.items()
    items += self.vpsets.items()
    if self.schedule:
        items += [("schedule", self.schedule)]
    return tuple(items)
cms.Process.items_=new_items_

from copy import deepcopy
import inspect

def new_Parameterizable_init(self,*a,**k):
  self.__dict__['_modifications'] = []
  self.old__init__(*a,**k)
  self._modifications = []
cms._Parameterizable.old__init__ = cms._Parameterizable.__init__
cms._Parameterizable.__init__ = new_Parameterizable_init

def new_Parameterizable_addParameter(self, name, value):
  self.old__addParameter(name,value)
  self._modifications.append({'file':inspect.stack()[3][1],'line':inspect.stack()[3][2],'name':name,'old':None,'new':deepcopy(value)})
cms._Parameterizable.old__addParameter = cms._Parameterizable._Parameterizable__addParameter
cms._Parameterizable._Parameterizable__addParameter = new_Parameterizable_addParameter

def new_Parameterizable_setattr(self, name, value):
  if (not self.isFrozen()) and (not name.startswith('_')) and (name in self.__dict__):
    self._modifications.append({'file':inspect.stack()[1][1],'line':inspect.stack()[1][2],'name':name,'old':deepcopy(self.__dict__[name]),'new':deepcopy(value)})
  self.old__setattr__(name,value)
cms._Parameterizable.old__setattr__ = cms._Parameterizable.__setattr__
cms._Parameterizable.__setattr__ = new_Parameterizable_setattr


    
  