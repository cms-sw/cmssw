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

def new___init__(self,*l,**k):
    self.old___init__(*l,**k)
    self.__dict__['_Process__history'] = []
    self.__dict__['_Process__enableRecording'] = 0
    self.__dict__['_Process__modifiedobjects'] = []
    self.__dict__['_Process__modifiedcheckpoint'] = None
    self.__dict__['_Process__modifications'] = []
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

def new_history(self, removeDuplicates=False):
    return self.__dict__['_Process__history']+self.dumpModificationsWithObjects(removeDuplicates)
cms.Process.history=new_history

def new_resetHistory(self):
    self.__dict__['_Process__history'] = []
    self.resetModified()
    self.resetModifiedObjects()
cms.Process.resetHistory=new_resetHistory

def new_dumpHistory(self,withImports=True):
    dumpHistory=[]
    for item,objects in self.history():
        if isinstance(item,(str,unicode)):
            dumpHistory.append(item +"\n")
        else: # isTool
	    print item
            dump=item.dumpPython()
            if isinstance(dump,tuple):
                if withImports and dump[0] not in dumpHistory:
                    dumpHistory.append(dump[0])
                dumpHistory.append(dump[1] +"\n")
            else:
                dumpHistory.append(dump +"\n")
           
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

def new_setattr(self, name, value):
    """
    This catches modifications that occur during process.load,
    and only records a modification if there was an existing object
    and the version after __setattr__ has a different id().
    This does not mean that the object is different, only redefined.
    We still really need a recursive-comparison function for parameterizeable
    objects to determine if a real change has been made.
    """
    old = None
    existing = False
    if not name.startswith('_Process__'):
        existing = hasattr(self, name)
        if existing:
            old = getattr(self, name)
    self.old__setattr__(name, value)
    if existing:
        if id(getattr(self, name)) != id(old):
            stack = auto_inspect()
            self.__dict__['_Process__modifications'] += [{'name': name,
                                                          'old': deepcopy(old), 
                                                          'new': deepcopy(getattr(self, name)),
                                                          'file':stack[0][1],'line':stack[0][2],
                                                          'action': 'replace'}]
cms.Process.old__setattr__ = cms.Process.__setattr__
cms.Process.__setattr__ = new_setattr

def new_recurseResetModified_(self, o):
    """
    Empty all the _modifications lists for
    all objects beneath this one.
    """
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

def new_recurseDumpModifications_(self, name, o):
    """
    Recursively return a standardised list of modifications
    from the object hierarchy.
    """
    modifications = []
    if isinstance(o, cms._ModuleSequenceType):
        if o._isModified:
            for mod in o._modifications:
                modifications.append({'name':name,
                                      'action':mod['action'],
                                      'old': mod['old'],
                                      'new': mod['new'],
                                      'file': mod['file'],
                                      'line': mod['line'],
                                      'dump': o.dumpPython({}),
                                      'type': 'seq'})
    
    if isinstance(o, cms._Parameterizable):
        for mod in o._modifications:
            paramname = mod['name']
            if hasattr(o, paramname):
                paramvalue = getattr(o, paramname)
            else:
                paramvalue = None
            if isinstance(paramvalue,cms._ParameterTypeBase):
                dump = paramvalue.dumpPython()
            else:
                dump = paramvalue
            modifications.append({'name': '%s.%s' %(name, paramname),
                                  'old': mod['old'],
                                  'new': mod['new'],
                                  'file': mod['file'],
                                  'line': mod['line'],
                                  'action': mod['action'],
                                  'dump': dump,
                                  'type': 'param'})
            
        # Loop over any child elements
        for key in o.parameterNames_():
            value = getattr(o,key)
            modifications += self.recurseDumpModifications_("%s.%s" % (name, key), value)
    
    if isinstance(o, cms._ValidatingListBase):
        for index, item in enumerate(o):
            modifications += self.recurseDumpModifications_("%s[%s]" % (name, index), item)
    if isinstance(o, cms.Process):
        for mod in o.__dict__['_Process__modifications']:
            if hasattr(o, mod['name']) and hasattr(getattr(o, mod['name']), 'dumpPython'):
                dump = getattr(o, mod['name']).dumpPython()
            else:
                dump = None
            modifications.append({'name': mod['name'],
                                  'action': mod['action'],
                                  'old': mod['old'],
                                  'new': mod['new'],
                                  'dump': dump,
                                  'file': mod['file'],
                                  'line': mod['line'],
                                  'type': 'process'})
    return modifications
cms.Process.recurseDumpModifications_=new_recurseDumpModifications_

def new_modificationCheckpoint(self):
    """
    Set a checkpoint, ie get the current list of all known
    top-level names and store them. Later, when we print out
    modifications we ignore any modifications that do not affect
    something in this list.

    There is currently no way of clearing this, but I think this
    is generally a use-once feature.
    """
    existing_names = set()
    for item in self.items_():
        existing_names.add(item[0])
    self.__dict__['_Process__modifiedcheckpoint'] = list(existing_names)
cms.Process.modificationCheckpoint=new_modificationCheckpoint

def new_resetModified(self):
    """
    Empty out all the modification lists, so we only see changes that
    happen from now onwards.
    """
    self.__dict__['_Process__modified'] = []
    for name, o in self.items_():
        self.recurseResetModified_(o)
cms.Process.resetModified=new_resetModified

def new_dumpModifications(self, comments=True, process=True, module=False, sequence=True, value=True, sort=True, group=True):
    """
    Return some text describing all the modifications that have been made.

    * comments: print out comments describing the file and line which triggered
                the modification, if determined.
    * process: print "process." in front of every name
    * module: only print out one entry per top-level module that has been
              changed, rather than the details
    * sequence: include changes to sequences
    * value: print out the latest value of each name
    * sort: whether to sort all the names before printing (otherwise they're in
            more-or-less time order, within each category)
    """
    modifications = self.recurseDumpModifications_('', self)
    text = []
    for name, o in self.items_():
        modifications += self.recurseDumpModifications_(name, o)
    if not sequence:
        modifications = filter(lambda x: not x['type'] == 'seq', modifications)
    checkpoint = self.__dict__['_Process__modifiedcheckpoint']
    if not checkpoint == None:
        modifications = filter(lambda x: any([x['name'].startswith(check) for check in checkpoint]), modifications)
    if module:
        value = False
        comments = False
        modules = list(set([m['name'].split('.')[0] for m in modifications]))
        if sort:
            modules = sorted(modules)
        if process:
            text = ['process.%s' % m for m in modules]
        else:
            text = modules
    else:
        if sort:
            modifications = sorted(modifications, key=lambda x: x['name'])
        for i, m in enumerate(modifications):
            t = ''
            if comments:
                if m['action'] == 'replace':
                    t += '# %(file)s:%(line)s replace %(old)s->%(new)s\n' % m
                elif m['action'] == 'remove':
                    t += '# %(file)s:%(line)s remove %(old)s\n' % m
                elif m['action'] == 'append':
                    t += '# %(file)s:%(line)s append %(new)s\n' % m
            if not group or i==len(modifications)-1 or not modifications[i+1]['name'] == m['name']:
                if process and value:
                    t += 'process.%s = %s' % (m['name'], m['dump'])
                elif value:
                    t += '%s = %s' % (m['name'], m['dump'])
                elif process:
                    t += 'process.%s' % (m['name'])
                else:
                    t += '%s' % (m['name'])
            text += [t]
    return '\n'.join(text)+'\n'
cms.Process.dumpModifications=new_dumpModifications

def new_dumpModificationsWithObjects(self, removeDuplicates=False):
    modifications = []
    last_modification=""
    for name, o in self.items_():
        for m in self.recurseDumpModifications_(name, o):
            # remove duplicate modifications
            if removeDuplicates and last_modification==m['name']:
                modifications.pop()
            last_modification=m['name']
            # add changes
            text = 'process.%s = %s' % (m['name'], m['dump'])
            modifications += [(text,[o])]
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
  self._modifications.append({'file':stack[0][1],'line':stack[0][2],'name':name,'old':None,'new':deepcopy(value),'action':'add'})
cms._Parameterizable.old__addParameter = cms._Parameterizable._Parameterizable__addParameter
cms._Parameterizable._Parameterizable__addParameter = new_Parameterizable_addParameter

def new_Parameterizable_setattr(self, name, value):
  if (not self.isFrozen()) and (not name.startswith('_')) and (name in self.__dict__):
    stack = auto_inspect()
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'name':name,'old':deepcopy(self.__dict__[name]),'new':deepcopy(value),'action':'replace'})
    self._isModified = True
  self.old__setattr__(name,value)
cms._Parameterizable.old__setattr__ = cms._Parameterizable.__setattr__
cms._Parameterizable.__setattr__ = new_Parameterizable_setattr

def new_Parameterizeable_delattr(self, name):
    if not self.isFrozen():
        stack = auto_inspect()
        self._modifications.append({'file':stack[0][1],'line':stack[0][2],'name':name,'old':deepcopy(self.__dict__[name]), 'new':None,'action':'delete'})
    self.old__delattr__(name)
cms._Parameterizable.old__delattr__ = cms._Parameterizable.__delattr__
cms._Parameterizable.__delattr__ = new_Parameterizeable_delattr


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
cms._Sequenceable._name_ = new__Sequenceable_name

try:
    # for backwards-compatibility with CMSSW_3_10_X
    from FWCore.ParameterSet.SequenceTypes import _SequenceOperator

    def new__SequenceOperator_name(self):
        return str(self._left._name_())+str(self._pySymbol)+str(self._right._name_())
    _SequenceOperator._name_ = new__SequenceOperator_name    
except:
    pass

from FWCore.ParameterSet.SequenceTypes import _SequenceNegation, _SequenceIgnore, SequencePlaceholder

def new__SequencePlaceholder_name(self):
    return self._name
SequencePlaceholder._name_ = new__SequencePlaceholder_name

def new__SequenceNegation_name(self):
    if self._operand:
        return '~'+str(self._operand._name_())
    else:
        return '~()'
_SequenceNegation._name_ = new__SequenceNegation_name    

def new__SequenceIgnore_name(self):
    if self._operand:
        return '-'+str(self._operand._name_())
    else:
        return '-()'
_SequenceIgnore._name_ = new__SequenceIgnore_name

def new_Sequence_name(self):
    if self._seq:
        return '('+str(self._seq._name_())+')'
    else:
        return '()'
cms.Sequence._name_ = new_Sequence_name

def new__Module_name(self):
  if hasattr(self,'_Labelable__label'):
    return getattr(self,'_Labelable__label')
  elif hasattr(self,'_TypedParameterizable__type'):
    return 'unnamed(%s)'%getattr(self,'_TypedParameterizable__type')
  return type(self).__name__
cms._Module._name_ = new__Module_name

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

def new__ModuleSequenceType_replace(self, original, replacement):
    stack = auto_inspect()
    self._isModified=True
    if replacement is None:
        self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'replace','old':original._name_(),'new':None})
    else:
        self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'replace','old':original._name_(),'new':replacement._name_()})
    return self.old_replace(original, replacement)
cms._ModuleSequenceType.old_replace = cms._ModuleSequenceType.replace
cms._ModuleSequenceType.replace = new__ModuleSequenceType_replace

def new__ModuleSequenceType_remove(self, original):
    stack = auto_inspect()
    self._isModified=True
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'remove','old':original._name_(),'new':None})
    return self.old_remove(original)
cms._ModuleSequenceType.old_remove = cms._ModuleSequenceType.remove
cms._ModuleSequenceType.remove = new__ModuleSequenceType_remove

def new__ModuleSequenceType__imul__(self,other):
    stack = auto_inspect()
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'append','new':other._name_(),'old':None})
    self._isModified=True
    return self.old__iadd__(other)
cms._ModuleSequenceType.old__imul__ = cms._ModuleSequenceType.__imul__
cms._ModuleSequenceType.__imul__ = new__ModuleSequenceType__imul__

def new__ModuleSequenceType__iadd__(self,other):
    stack = auto_inspect()
    self._isModified=True
    self._modifications.append({'file':stack[0][1],'line':stack[0][2],'action':'append','new':other._name_(),'old':None})
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
            self.assert_('process.ex.nine' in mods)
            self.assert_('process.ex.newvpset' in mods)
            
            

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
    
            self.assertEqual(process.dumpHistory(),"\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process , 'file:filename.root')\n\n\nchangeSource(process , 'file:filename2.root')\n\n\nchangeSource(process , 'file:filename3.root')\n\n")
            
            process.source.fileNames=cms.untracked.vstring("file:replacedfile.root") 
            self.assertEqual(process.dumpHistory(),"\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process , 'file:filename.root')\n\n\nchangeSource(process , 'file:filename2.root')\n\n\nchangeSource(process , 'file:filename3.root')\n\nprocess.source.fileNames = cms.untracked.vstring('file:replacedfile.root')\n")
            
            process.disableRecording()
            changeSource.setParameter('source',"file:filename4.root")
            action=changeSource.__copy__()
            process.addAction(action)
            self.assertEqual(process.dumpHistory(),"\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process , 'file:filename.root')\n\n\nchangeSource(process , 'file:filename2.root')\n\n\nchangeSource(process , 'file:filename3.root')\n\nprocess.source.fileNames = cms.untracked.vstring('file:replacedfile.root')\n")
            
            process.enableRecording()
            changeSource.setParameter('source',"file:filename5.root")
            action=changeSource.__copy__()
            process.addAction(action)
            process.deleteAction(3)
            self.assertEqual(process.dumpHistory(),"\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process , 'file:filename.root')\n\n\nchangeSource(process , 'file:filename2.root')\n\n\nchangeSource(process , 'file:filename3.root')\n\n\nchangeSource(process , 'file:filename5.root')\n\n")

            process.deleteAction(0)
            self.assertEqual(process.dumpHistory(),"\nfrom FWCore.GuiBrowsers.editorTools import *\n\nchangeSource(process , 'file:filename2.root')\n\n\nchangeSource(process , 'file:filename3.root')\n\n\nchangeSource(process , 'file:filename5.root')\n\n")
            
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
        
