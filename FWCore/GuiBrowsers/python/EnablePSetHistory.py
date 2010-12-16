from copy import deepcopy
import inspect

ACTIVATE_INSPECTION=True

#### helpers for inspection ####

def auto_inspect():
    if not ACTIVATE_INSPECTION:
        return [("unknown","unknown","unknown")]
    stack = inspect.stack()
    while len(stack)>=1 and len(stack[0])>=2 and ('FWCore/ParameterSet' in stack[0][1] or 'FWCore/GuiBrowsers' in stack[0][1]):
        stack = stack[1:]
    if len(stack)>=1 and len(stack[0])>=3:
       return stack
    else:
       return [("unknown","unknown","unknown")]

#### patches needed for deepcopy of process ####

import FWCore.ParameterSet.DictTypes as typ
    
def new_SortedKeysDict__copy__(self):
    return self.__class__(self)
typ.SortedKeysDict.__copy__ = new_SortedKeysDict__copy__

def new_SortedKeysDict__deepcopy__(self, memo=None):
    from copy import deepcopy
    if memo is None:
        memo = {}
    d = memo.get(id(self), None)
    if d is not None:
        return d
    memo[id(self)] = d = self.__class__()
    d.__init__(deepcopy(self.items(), memo))
    return d
typ.SortedKeysDict.__deepcopy__ = new_SortedKeysDict__deepcopy__

#### process history ####

import FWCore.ParameterSet.Config as cms

def new___init__(self,name):
    self.old___init__(name)
    self.__dict__['_Process__history'] = []
    self.__dict__['_Process__enableRecording'] = 0
    self.__dict__['_Process__modifiedobjects'] = []
cms.Process.old___init__=cms.Process.__init__
cms.Process.__init__=new___init__

def new_modifiedObjects(self):
    return self.__dict__['_Process__modifiedobjects']
cms.Process.modifiedObjects=new_modifiedObjects

def new_resetModifiedObjects(self):
    self.__dict__['_Process__modifiedobjects'] = []
cms.Process.resetModifiedObjects=new_resetModifiedObjects

def new__place(self, name, mod, d):
    self.old__place(name, mod, d)
    if self._okToPlace(name, mod, d):
        self.__dict__['_Process__modifiedobjects'].append(mod)
cms.Process.old__place=cms.Process._place
cms.Process._place=new__place

def new__placeSource(self, name, mod):
    self.old__placeSource(name, mod)
    self.__dict__['_Process__modifiedobjects'].append(mod)
cms.Process.old__placeSource=cms.Process._placeSource
cms.Process._placeSource=new__placeSource

def new__placeLooper(self, name, mod):
    self.old__placeLooper(name, mod)
    self.__dict__['_Process__modifiedobjects'].append(mod)
cms.Process.old__placeLooper=cms.Process._placeLooper
cms.Process._placeLooper=new__placeLooper

def new__placeService(self, typeName, mod):
    self.old__placeService(typeName, mod)
    self.__dict__['_Process__modifiedobjects'].append(mod)
cms.Process.old__placeService=cms.Process._placeService
cms.Process._placeService=new__placeService

def new_setSchedule_(self, sch):
    self.old_setSchedule_(sch)
    self.__dict__['_Process__modifiedobjects'].append(sch)
cms.Process.old_setSchedule_=cms.Process.setSchedule_
cms.Process.setSchedule_=new_setSchedule_

def new_setLooper_(self, lpr):
    self.old_setLooper_(lpr)
    self.__dict__['_Process__modifiedobjects'].append(lpr)
cms.Process.old_setLooper_=cms.Process.setLooper_
cms.Process.setLooper_=new_setLooper_

def new_history(self):
    return self.__dict__['_Process__history']+self.dumpModificationsWithObjects()
cms.Process.history=new_history

def new_resetHistory(self):
    self.__dict__['_Process__history'] = []
    self.resetModified()
    self.resetModifiedObjects()
cms.Process.resetHistory=new_resetHistory

def new_dumpHistory(self,withImports=True):
    dumpHistory=[]
    for item,objects in self.history():
        if isinstance(item,str):
            dumpHistory.append(item +"\n")
        else: # isTool
            dump=item.dumpPython()
            if isinstance(dump,tuple):
                if withImports and dump[0] not in dumpHistory:
                    dumpHistory.append(dump[0])
                dumpHistory.append(dump[1])
            else:
                dumpHistory.append(dump)
           
    return ''.join(dumpHistory)
cms.Process.dumpHistory=new_dumpHistory

def new_addAction(self,tool):
    if self.__dict__['_Process__enableRecording'] == 0:
        modifiedObjects=self.modifiedObjects()
        for m,o in self.dumpModificationsWithObjects():
            modifiedObjects+=o
        self.__dict__['_Process__history'].append((tool,modifiedObjects))
        self.resetModified()
        self.resetModifiedObjects()
cms.Process.addAction=new_addAction

def new_deleteAction(self,i):
    del self.__dict__['_Process__history'][i]
cms.Process.deleteAction=new_deleteAction

def new_disableRecording(self):
    if self.__dict__['_Process__enableRecording'] == 0:
        # remember modifications in history
        self.__dict__['_Process__history']+=self.dumpModificationsWithObjects()
        self.resetModified()
        self.resetModifiedObjects()
    self.__dict__['_Process__enableRecording'] += 1
cms.Process.disableRecording=new_disableRecording

def new_enableRecording(self):
    self.__dict__['_Process__enableRecording'] -= 1
cms.Process.enableRecording=new_enableRecording

def new_checkRecording(self):
    return self.__dict__['_Process__enableRecording']==0
cms.Process.checkRecording=new_checkRecording

def new_recurseResetModified_(self, o):
    properties = []
    if isinstance(o, cms._ModuleSequenceType):
        o.resetModified()
    if isinstance(o, cms._Parameterizable):
        o.resetModified()
        for key in o.parameterNames_():
            value = getattr(o,key)
            self.recurseResetModified_(value)
    if isinstance(o, cms._ValidatingListBase):
        for index,item in enumerate(o):
            self.recurseResetModified_(item)
cms.Process.recurseResetModified_=new_recurseResetModified_

def new_recurseDumpModifiedShort_(self, name, o, allow_seq):
    names = set()
    if isinstance(o, cms._ModuleSequenceType) and allow_seq:
        if o._isModified:
            names.add(name)
    if isinstance(o, cms._Parameterizable):
        for key in o.parameterNames_():
            value = getattr(o,key)
            names |= self.recurseDumpModifiedShort_("%s.%s"%(name,key),value,allow_seq)
        for mod in o._modifications:
            names.add('%s.%s' % (name, mod['name']))
    if isinstance(o, cms._ValidatingListBase):
        for index, item in enumerate(o):
            names |= self.recurseDumpModifiedShort_("%s[%s]"%(name,index),item,allow_seq)
    return names
cms.Process.recurseDumpModifiedShort_=new_recurseDumpModifiedShort_

def new_recurseDumpModifications_(self, name, o, comments=True):
    dumpPython = ""
    if isinstance(o, cms._ModuleSequenceType):
        if o._isModified:
            if dumpPython != "":
                dumpPython += "\n"
            if comments:
                for mod in o._modifications:
                    if mod['action']=='replace':
                        dumpPython += "# MODIFIED BY %(file)s:%(line)s replace %(old)s with %(new)s\n"%mod
                    if mod['action']=='remove':
                        dumpPython += "# MODIFIED BY %(file)s:%(line)s remove %(old)s\n"%mod
                    if mod['action']=='append':
                        dumpPython += "# MODIFIED BY %(file)s:%(line)s append %(new)s\n"%mod
            dumpPython += "process.%s = %s"%(name,o.dumpPython({}))
    
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
            if dumpPython != "":
                dumpPython += "\n"
            if comments:
                for mod in mod_dict[paramname]:
                    if isinstance(mod['old'],cms._ParameterTypeBase):
                      mod['old'] = mod['old'].dumpPython().replace('\n','')
                    if isinstance(mod['new'],cms._ParameterTypeBase):
                      mod['new'] = mod['new'].dumpPython().replace('\n','')
                    dumpPython += "# MODIFIED BY %(file)s:%(line)s; %(old)s -> %(new)s\n" % mod
            paramvalue = getattr(o,paramname)
            if isinstance(paramvalue,cms._ParameterTypeBase):
              paramvalue = paramvalue.dumpPython().replace('\n','')
            dumpPython += "process.%s.%s = %s\n" % (name,paramname,paramvalue) # Currently, _Parameterizable doesn't check __delattr__ for modifications. We don't either, but if anyone does __delattr__ then this will fail.
            
        # Loop over any child elements
        for key in o.parameterNames_():
            value = getattr(o,key)
            dumpPython += self.recurseDumpModifications_("%s.%s"%(name,key),value,comments)
    
    # Test if we have a VPSet (I think the code above would miss checking a VPSet for modified children too)
    if isinstance(o, cms._ValidatingListBase):
        for index,item in enumerate(o):
            dumpPython += self.recurseDumpModifications_("%s[%s]"%(name,index),item,comments)
    return dumpPython    
cms.Process.recurseDumpModifications_=new_recurseDumpModifications_

def new_resetModified(self):
    for name, o in self.items_():
        self.recurseResetModified_(o)
cms.Process.resetModified=new_resetModified

def new_dumpModifications(self,comments=True):
    dumpModifications = ""
    for name, o in self.items_():
        dumpPython = self.recurseDumpModifications_(name, o, comments)
        if dumpPython != "":
            if dumpModifications != "":
                dumpModifications += "\n"
            dumpModifications += dumpPython
    return dumpModifications
cms.Process.dumpModifications=new_dumpModifications

def new_dumpModificationsShort(self, module=True, process=True, sequence=True):
    names = set()
    for name, o in self.items_():
        names |= self.recurseDumpModifiedShort_(name, o, sequence)
    if module:
        names = set([n.split('.')[0] for n in names])
    if process:
        return '\n'.join(['process.%s'%n for n in sorted(list(names))])
    else:
        return '\n'.join(sorted(list(names)))
cms.Process.dumpModificationsShort=new_dumpModificationsShort

def new_dumpModificationsWithObjects(self):
    modifications = []
    for name, o in self.items_():
        for m in self.recurseDumpModifications_(name, o, False).split("\n"):
            if m != "":
                modifications += [(m,[o])]
    return modifications
cms.Process.dumpModificationsWithObjects=new_dumpModificationsWithObjects

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

#### parameterizable history ####

def new_Parameterizable_init(self,*a,**k):
  self.__dict__['_modifications'] = []
  self.old__init__(*a,**k)
  self._modifications = []
cms._Parameterizable.old__init__ = cms._Parameterizable.__init__
cms._Parameterizable.__init__ = new_Parameterizable_init

def new_Parameterizable_addParameter(self, name, value):
  self.old__addParameter(name,value)
  stack = auto_inspect()
  self._modifications.append({'file':stack[0][1],'line':stack[0][2],'name':name,'old':None,'new':deepcopy(value)})
cms._Parameterizable.old__addParameter = cms._Parameterizable._Parameterizable__addParameter
cms._Parameterizable._Parameterizable__addParameter = new_Parameterizable_addParameter

def new_Parameterizable_setattr(self, name, value):
  if (not self.isFrozen()) and (not name.startswith('_')) and (name in self.__dict__):
    stack = auto_inspect()
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'name':name,'old':deepcopy(self.__dict__[name]),'new':deepcopy(value)})
    self._isModified = True
  self.old__setattr__(name,value)
cms._Parameterizable.old__setattr__ = cms._Parameterizable.__setattr__
cms._Parameterizable.__setattr__ = new_Parameterizable_setattr

def new_Parameterizable_resetModified(self):
    self._isModified=False
    self._modifications = []
    for name in self.parameterNames_():
        param = self.__dict__[name]
        if isinstance(param, cms._Parameterizable):
            param.resetModified()
cms._Parameterizable.resetModified = new_Parameterizable_resetModified

def new_ParameterTypeBase_resetModified(self):
    self._isModified=False
    self._modifications = []
cms._ParameterTypeBase.resetModified = new_ParameterTypeBase_resetModified

#### sequence history ####

def new__Sequenceable_name(self):
    return ''
cms._Sequenceable._name = new__Sequenceable_name

from FWCore.ParameterSet.SequenceTypes import _SequenceOperator, _SequenceNegation, _SequenceIgnore


def new__SequenceOperator_name(self):
    return str(self._left._name())+str(self._pySymbol)+str(self._right._name())
_SequenceOperator._name = new__SequenceOperator_name    

def new__SequenceNegation_name(self):
    return '~'+str(self._operand._name())
_SequenceNegation._name = new__SequenceNegation_name    

def new__SequenceIgnore_name(self):
    return '-'+str(self._operand._name())
_SequenceIgnore._name = new__SequenceIgnore_name

def new_Sequence_name(self):
    return '('+str(self._seq._name())+')'
cms.Sequence._name = new_Sequence_name

def new__Module_name(self):
  if hasattr(self,'_Labelable__label'):
    return getattr(self,'_Labelable__label')
  elif hasattr(self,'_TypedParameterizable__type'):
    return 'unnamed(%s)'%getattr(self,'_TypedParameterizable__type')
  return type(self).__name__
cms._Module._name = new__Module_name

def new__ModuleSequenceType__init__(self,*arg,**argv):
    self._modifications = []
    self.old__init__(*arg,**argv)
cms._ModuleSequenceType.old__init__ = cms._ModuleSequenceType.__init__
cms._ModuleSequenceType.__init__ = new__ModuleSequenceType__init__
    
def new__ModuleSequenceType_resetModified(self):
    self._isModified=False
    self._modifications = []
cms._ModuleSequenceType.resetModified = new__ModuleSequenceType_resetModified

def new__ModuleSequenceType_isModified(self):
    return self._isModified
cms._ModuleSequenceType.isModified = new__ModuleSequenceType_isModified

def new__ModuleSequenceType_copy(self):
    returnValue = cms._ModuleSequenceType.__new__(type(self))
    returnValue.__init__(self._seq)
    returnValue._isModified = self._isModified
    returnValue._modifications = deepcopy(self._modifications)
    return returnValue
cms._ModuleSequenceType.copy = new__ModuleSequenceType_copy

def new__ModuleSequenceType__replace(self, original, replacement):
    stack = auto_inspect()
    self._isModified=True
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'replace','old':original._name(),'new':replacement._name()})
    self.old__replace(original, replacement)
cms._ModuleSequenceType.old__replace = cms._ModuleSequenceType._replace
cms._ModuleSequenceType._replace = new__ModuleSequenceType__replace

def new__ModuleSequenceType__remove(self, original):
    stack = auto_inspect()
    self._isModified=True
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'remove','old':original._name()})
    return self.old__remove(original)
cms._ModuleSequenceType.old__remove = cms._ModuleSequenceType._remove
cms._ModuleSequenceType._remove = new__ModuleSequenceType__remove

def new__ModuleSequenceType__imul__(self,other):
    stack = auto_inspect()
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'append','new':other._name()})
    self._isModified=True
    return self.old__iadd__(other)
cms._ModuleSequenceType.old__imul__ = cms._ModuleSequenceType.__imul__
cms._ModuleSequenceType.__imul__ = new__ModuleSequenceType__imul__

def new__ModuleSequenceType__iadd__(self,other):
    stack = auto_inspect()
    self._isModified=True
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'append','new':other._name()})
    return self.old__iadd__(other)
cms._ModuleSequenceType.old__iadd__ = cms._ModuleSequenceType.__iadd__
cms._ModuleSequenceType.__iadd__ = new__ModuleSequenceType__iadd__

from FWCore.ParameterSet.Modules  import Source
from FWCore.GuiBrowsers.editorTools import changeSource
            
if __name__=='__main__':
    import unittest
    class TestModificationTracking(unittest.TestCase):
        def setUp(self):
            pass
        def testPSet(self):
            ex = cms.EDAnalyzer("Example",
                one = cms.double(0),
                two = cms.bool(True),
                ps = cms.PSet(
                    three = cms.int32(10),
                    four = cms.string('abc')
                ),
                vps = cms.VPSet(
                    cms.PSet(
                        five = cms.InputTag('alpha')
                    ),
                    cms.PSet(
                        six = cms.vint32(1,2,3)
                    )
                ),
                seven = cms.vstring('alpha','bravo','charlie'),
                eight = cms.vuint32(range(10)),
                nine = cms.int32(0)
            )
            ex.zero = cms.string('hello')
            self.assertEqual(ex._modifications[-1]['name'],'zero')
            ex.one = cms.double(1)
            ex.one = cms.double(2)
            ex.one = cms.double(3)
            self.assertEqual(ex._modifications[-1]['name'],'one')
            self.assertEqual(ex._modifications[-2]['name'],'one')
            self.assertEqual(ex._modifications[-3]['name'],'one')
            ex.two = False
            self.assertEqual(ex._modifications[-1]['name'],'two')
            ex.ps.three.setValue(100) # MISSED
            #self.assertEqual(ex.ps._modifications.pop()['name'],'three')
            ex.ps.four = 'def'
            self.assertEqual(ex.ps._modifications[-1]['name'],'four')
            ex.vps[0].five = cms.string('beta')
            self.assertEqual(ex.vps[0]._modifications[-1]['name'],'five')
            ex.vps[1].__dict__['six'] = cms.vint32(1,4,9) # MISSED
            #self.assertEqual(ex.vps[1]._modifications[-1]['name'],'six')
            ex.seven[0] = 'delta' # MISSED
            #self.assertEqual(ex._modifications[-1]['name'],'seven')
            ex.eight.pop() # MISSED
            #self.assertEqual(ex._modifications[-1]['name'],'eight')
            del ex.nine
            #self.assertEqual(ex._modifications[-1]['name'],'nine')
            ex.newvpset = cms.VPSet()
            self.assertEqual(ex._modifications[-1]['name'],'newvpset')
            
            process = cms.Process('unittest')
            process.ex = ex
            mods = process.dumpModifications()
            self.assert_('process.ex.zero' in mods)
            self.assert_('process.ex.one' in mods)
            self.assert_('process.ex.two' in mods)
            #self.assert_('process.ex.three' in mods)
            self.assert_('process.ex.ps.four' in mods)
            self.assert_('process.ex.vps[0].five' in mods)
            #self.assert_('process.ex.vps[1].six' in mods)
            #self.assert_('process.ex.seven[0]' in mods)
            #self.assert_('process.ex.eight' in mods)
            #self.assert_('process.ex.nine' in mods)
            self.assert_('process.ex.newvpset' in mods)
        
        def testShort(self):
            process = cms.Process('unittest')
            for i in range(10):
                setattr(process, 'f%s'%i, cms.EDFilter('f%s'%i, **dict([(c, cms.string(c)) for c in 'abcdef'])))
            process.seq1 = cms.Sequence(process.f0*process.f1*process.f2)
            process.seq2 = cms.Sequence(process.f3*process.f4*process.f5)
            
            self.assertEqual(process.dumpModificationsShort(), '')
            
            process.seq1.replace(process.f0, process.f6*process.f7)
            
            self.assert_('process.seq1' in process.dumpModificationsShort())
            
            process.f8.a = cms.string('b')
            
            self.assert_('process.f8' in process.dumpModificationsShort())
            self.assert_('process.f8.a' in process.dumpModificationsShort(False))
            
            

        def testSeq(self):
            process = cms.Process('unittest')
            for i in range(10):
              setattr(process,'f%s'%i,cms.EDFilter('f%s'%i))
            process.seq1 = cms.Sequence(process.f1*process.f2*process.f3)
            self.assertEqual(process.seq1._modifications,[])
            process.seq2 = cms.Sequence(process.f4+process.f5+process.f6)
            self.assertEqual(process.seq2._modifications,[])
            
            process.seq1.replace(process.f1,process.f0*process.f1)
            self.assertEqual(process.seq1._modifications[-1]['action'],'replace')
            
            process.seq2.remove(process.f5)
            self.assertEqual(process.seq2._modifications[-1]['action'],'remove')
            
            process.path = cms.Path(process.seq1*process.f7)
            self.assertEqual(process.path._modifications,[])
            
            process.path *= process.seq2
            self.assertEqual(process.path._modifications[-1]['action'],'append')
            process.path.remove(process.f6)
            self.assertEqual(process.path._modifications[-1]['action'],'remove')
            process.path.replace(process.f2,~process.f2)
            self.assertEqual(process.path._modifications[-1]['action'],'replace')
            
            mods = process.dumpModifications()
            self.assert_('process.seq1' in mods)
            self.assert_('process.seq2' in mods)
            self.assert_('process.path' in mods)
            
        def testdumpHistory(self):
            process = cms.Process('unittest')
            process.source=Source("PoolSource",fileNames = cms.untracked.string("file:file.root"))
            
            changeSource(process,"file:filename.root")
            self.assertEqual(changeSource._parameters['source'].value,"file:filename.root")
            
            changeSource(process,"file:filename2.root")
            self.assertEqual(changeSource._parameters['source'].value,"file:filename2.root")
            
            changeSource(process,"file:filename3.root")
            self.assertEqual(changeSource._parameters['source'].value,"file:filename3.root")
    
            self.assertEqual(process.dumpHistory(),'\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process, "file:filename.root")\n\nchangeSource(process, "file:filename2.root")\n\nchangeSource(process, "file:filename3.root")\n')
            
            process.source.fileNames=cms.untracked.vstring("file:replacedfile.root") 
            self.assertEqual(process.dumpHistory(),'\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process, "file:filename.root")\n\nchangeSource(process, "file:filename2.root")\n\nchangeSource(process, "file:filename3.root")\nprocess.source.fileNames = cms.untracked.vstring(\'file:replacedfile.root\')\n')
            
            process.disableRecording()
            changeSource.setParameter('source',"file:filename4.root")
            action=changeSource.__copy__()
            process.addAction(action)
            self.assertEqual(process.dumpHistory(),'\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process, "file:filename.root")\n\nchangeSource(process, "file:filename2.root")\n\nchangeSource(process, "file:filename3.root")\nprocess.source.fileNames = cms.untracked.vstring(\'file:replacedfile.root\')\n')
            
            process.enableRecording()
            changeSource.setParameter('source',"file:filename5.root")
            action=changeSource.__copy__()
            process.addAction(action)
            process.deleteAction(3)
            self.assertEqual(process.dumpHistory(),'\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process, "file:filename.root")\n\nchangeSource(process, "file:filename2.root")\n\nchangeSource(process, "file:filename3.root")\n\nchangeSource(process, "file:filename5.root")\n')

            process.deleteAction(0)
            self.assertEqual(process.dumpHistory(),'\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process, "file:filename2.root")\n\nchangeSource(process, "file:filename3.root")\n\nchangeSource(process, "file:filename5.root")\n')
            
        def testModifiedObjectsHistory(self):
            process = cms.Process('unittest')
            process.source=Source("PoolSource",fileNames = cms.untracked.string("file:file.root"))
            
            changeSource(process,"file:filename.root")
            self.assertEqual(len(process.history()[0][1]),1)
            
            process.source.fileNames=cms.untracked.vstring("file:replacedfile.root") 
            self.assertEqual(len(process.history()[0][1]),1)
            self.assertEqual(len(process.history()[1][1]),1)

            process.source.fileNames=["test2"]
            self.assertEqual(len(process.history()[0][1]),1)
            self.assertEqual(len(process.history()[1][1]),1)

            changeSource(process,"file:filename2.root")
            self.assertEqual(len(process.history()[0][1]),1)
            self.assertEqual(len(process.history()[1][1]),1)
            self.assertEqual(len(process.history()[2][1]),1)
            
            process.source.fileNames=cms.untracked.vstring("file:replacedfile2.root") 
            self.assertEqual(len(process.history()[0][1]),1)
            self.assertEqual(len(process.history()[1][1]),1)
            self.assertEqual(len(process.history()[2][1]),1)
            self.assertEqual(len(process.history()[3][1]),1)
            
    unittest.main()
        
