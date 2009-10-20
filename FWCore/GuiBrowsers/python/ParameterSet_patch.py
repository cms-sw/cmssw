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
