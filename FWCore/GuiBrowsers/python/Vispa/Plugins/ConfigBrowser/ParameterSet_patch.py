import FWCore.ParameterSet.DictTypes as dic

def patch_SortedKeysDict_items(self):
    return [(key, dict.__getitem__(self, key)) for key in self.list]
setattr(dic.SortedKeysDict,"items",patch_SortedKeysDict_items)

import FWCore.ParameterSet.Mixins as mix

def patch_ParameterTypeBase___init__(self):
    self.__dict__["_isFrozen"] = False
    self._ParameterTypeBase__isTracked = True
    self._isModified = False
def patch_ParameterTypeBase_isModified(self):
    return self._isModified
def patch_ParameterTypeBase_resetModified(self):
    self._isModified=False
setattr(mix._ParameterTypeBase,"__init__",patch_ParameterTypeBase___init__)
setattr(mix._ParameterTypeBase,"isModified",patch_ParameterTypeBase_isModified)
setattr(mix._ParameterTypeBase,"resetModified",patch_ParameterTypeBase_resetModified)

def patch_SimpleParameterTypeBase_setValue(self,value):
    if not self._isValid(value):
        raise ValueError(str(value)+" is not a valid "+str(type(self)))
    if value!=self._value:
        self._isModified=True
        self._value=value
setattr(mix._SimpleParameterTypeBase,"setValue",patch_SimpleParameterTypeBase_setValue)

def patch_ValidatingParameterListBase_setValue(self,v):
    self[:] = []
    self.extend(v)
    self._isModified=True
setattr(mix._ValidatingParameterListBase,"setValue",patch_ValidatingParameterListBase_setValue)

import FWCore.ParameterSet.Types as typ

def patch_InputTag_setModuleLabel(self,label):
    if label!=self._InputTag__moduleLabel:
        self._isModified=True
        self._InputTag__moduleLabel=label
def patch_InputTag_setProductInstanceLabel(self,label):
    if label!=self._InputTag__productInstance:
        self._isModified=True
        self._InputTag__productInstance=label
def patch_InputTag_setProcessName(self,label):
    if label!=self._InputTag__processName:
        self._isModified=True
        self._InputTag__processName=label
def patch_InputTag_setValue(self,v):
    if isinstance(v,tuple):
        self._setValues(*v)
    else:
        self._setValues(v)
    self._isModified=True
setattr(typ.InputTag,"setModuleLabel",patch_InputTag_setModuleLabel)
setattr(typ.InputTag,"setProductInstanceLabel",patch_InputTag_setProductInstanceLabel)
setattr(typ.InputTag,"setProcessName",patch_InputTag_setProcessName)
setattr(typ.InputTag,"setValue",patch_InputTag_setValue)

def patch_ESInputTag_setModuleLabel(self,label):
    if label!=self._ESInputTag__moduleLabel:
        self._isModified=True
        self._ESInputTag__moduleLabel=label
def patch_ESInputTag_setDataLabel(self,label):
    if label!=self._ESInputTag__data:
        self._isModified=True
        self._ESInputTag__data=label
def patch_ESInputTag_setValue(self,v):
    self._isModified=True
    self._setValues(v)
# ESInputTag moved to different file
#setattr(typ.ESInputTag,"setModuleLabel",patch_ESInputTag_setModuleLabel)
#setattr(typ.ESInputTag,"setDataLabel",patch_ESInputTag_setDataLabel)
#setattr(typ.ESInputTag,"setValue=patch",ESInputTag_setValue)

import FWCore.ParameterSet.Config as cms

def patch_Process_recurseResetModified_(self, object):
    properties=[]
    if hasattr(object,"parameterNames_"):
        for key in object.parameterNames_():
            value=getattr(object,key)
            if hasattr(value,"resetModified"):
                value.resetModified()
            self.recurseResetModified_(value)
def patch_Process_recurseDumpModified_(self, name, object):
    dumpPython=""
    if hasattr(object,"parameterNames_"):
        for key in object.parameterNames_():
            value=getattr(object,key)
            if not isinstance(value,typ.PSet):
                if hasattr(value,"isModified") and getattr(object,key).isModified():
                    if isinstance(value, cms.InputTag):
                        pythonValue = "\""+str(value.value())+"\""
                    elif hasattr(value,"pythonValue"):
                        pythonValue=value.pythonValue()
                    elif hasattr(value,"value"):
                        pythonValue=value.value()
                    else:
                        pythonValue=value
                    dumpPython+="process."+name+"."+key+" = "+str(pythonValue)+"\n"
            dumpPython+=self.recurseDumpModified_(name+"."+key,value)
    return dumpPython
def patch_Process_resetModified(self):
    for name, object in self.items_():
        self.recurseResetModified_(object)
def patch_Process_dumpModified(self):
    dumpPython=""
    for name, object in self.items_():
        dumpPython+=self.recurseDumpModified_(name, object)
    return dumpPython
def patch_Process_moduleItems_(self):
    items=[]
    items+=self.producers.items()
    items+=self.filters.items()
    items+=self.analyzers.items()
    return tuple(items)
def patch_Process_items_(self):
    items=[]
    if self.source:
        items+=[("source",self.source)]
    if self.looper:
        items+=[("looper",self.looper)]
    items+=self.moduleItems_()
    items+=self.outputModules.items()
    items+=self.sequences.items()
    items+=self.paths.iteritems()
    items+=self.endpaths.items()
    items+=self.services.items()
    items+=self.es_producers.items()
    items+=self.es_sources.items()
    items+=self.es_prefers.items()
    items+=self.psets.items()
    items+=self.vpsets.items()
    if self.schedule:
        items+=[("schedule",self.schedule)]
    return tuple(items)

setattr(cms.Process,"recurseResetModified_",patch_Process_recurseResetModified_)
setattr(cms.Process,"recurseDumpModified_",patch_Process_recurseDumpModified_)
setattr(cms.Process,"resetModified",patch_Process_resetModified)
setattr(cms.Process,"dumpModified",patch_Process_dumpModified)
setattr(cms.Process,"moduleItems_",patch_Process_moduleItems_)
setattr(cms.Process,"items_",patch_Process_items_)

if __name__ == "__main__":
    import unittest
    class TestPatchParamterTypeBase(unittest.TestCase):
        def testModified(self):
            class TestType(mix._SimpleParameterTypeBase):
                def _isValid(self,value):
                    return True
            a=TestType(1)
            self.assertEqual(a.isModified(),False)
            a.setValue(1)
            self.assertEqual(a.isModified(),False)
            a.setValue(2)
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            self.assertEqual(a.isModified(),False)
    class TestPatchInputTag(unittest.TestCase):
        def testModified(self):
            a=typ.InputTag("a")
            self.assertEqual(a.isModified(),False)
            a.setModuleLabel("a")
            self.assertEqual(a.isModified(),False)
            a.setModuleLabel("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setProductInstanceLabel("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setProcessName("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setValue("b")
            self.assertEqual(a.isModified(),True)
    class TestPatchProcess(unittest.TestCase):
        def testModified(self):
            process=cms.Process("a")
            process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('a'))
            self.assertEqual(process.dumpModified(),"")
            process.source.fileNames=['b']
            self.assertEqual(process.dumpModified(),"process.source.fileNames = ['b']\n")
            process.resetModified()
            self.assertEqual(process.dumpModified(),"")
    
    unittest.main()
