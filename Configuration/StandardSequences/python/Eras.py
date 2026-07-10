import FWCore.ParameterSet.Config as cms
from  FWCore.ParameterSet.Config import ModifierChain,Modifier
import Configuration.Eras
import pkgutil

class Eras (object):
    """
    Dummy container for all the cms.Modifier instances that config fragments
    can use to selectively configure depending on what scenario is active.
    """
    def __init__(self):
        allModules = [name for _, name, _ in pkgutil.iter_modules(Configuration.Eras.__path__)]
        prefixes = ['Era_', 'Modifier_', 'ModifierChain_']
        allEras = [m for m in allModules if any([m.startswith(p) for p in prefixes])]

        self.pythonCfgLines = {}

        def importEraObj(modName):
            import importlib
            import re
            p_pattern = '|'.join(prefixes)
            s_pattern = '|'.join(['_cff'])
            eraName = re.match(rf'^(?:{p_pattern})(.*)(?:{s_pattern})$', modName).group(1)
            eraObj = getattr(importlib.import_module(f'Configuration.Eras.{modName}'), eraName)
            return eraName, eraObj

        for e in allEras:
            eName, eObj = importEraObj(e)
            self.addEra(eName,eObj)
            self.pythonCfgLines[eName] = f'from Configuration.Eras.{e} import {eName}'

    def addEra(self,name,obj):
        setattr(self,name,obj)

    def inspectModifier(self,m,details):
        print('      ',m.__dict__ ['_Modifier__processModifiers'])

    def inspectEra(self,e,details):
        print('\nEra:',e)
        print('   isChosen:',getattr(self,e)._isChosen())
        if details: print('   Modifiers:')
        nmod=0
        for value in getattr(self,e).__dict__['_ModifierChain__chain']:
            if isinstance(value, Modifier):
                nmod=nmod+1
                if details: self.inspectModifier(value,details)
        print('   ',nmod,'modifiers defined')

    def inspect(self,name=None,onlyChosen=False,details=True):
        if name==None:
            print('Inspecting the known eras', end=' ')
            if onlyChosen: print(' (all active)')
            else: print('(all eras defined)')
        else:
            print('Inspecting the '+name+' era', end=' ')

        allEras=[]
        for key, value in self.__dict__.items():
            if isinstance(value, ModifierChain): allEras.append(key)

        for e in allEras:
            if name is not None and name==e:
                self.inspectEra(e,details)
            if name is None:
                if not onlyChosen or getattr(self,e).isChosen():
                    self.inspectEra(e,details)

eras=Eras()


#eras.inspect()
