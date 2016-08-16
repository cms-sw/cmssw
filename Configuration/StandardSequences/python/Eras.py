import FWCore.ParameterSet.Config as cms
from  FWCore.ParameterSet.Config import ModifierChain,Modifier

class Eras (object):
    """
    Dummy container for all the cms.Modifier instances that config fragments
    can use to selectively configure depending on what scenario is active.
    """
    def addEra(self,name,obj):
        setattr(self,name,obj)

    def inspectModifier(self,m,details):
        print '      ',m.__dict__ ['_Modifier__processModifiers']

    def inspectEra(self,e,details):
        print '\nEra:',e
        print '   isChosen:',getattr(self,e).isChosen()
        if details: print '   Modifiers:'
        nmod=0
        for value in getattr(self,e).__dict__['_ModifierChain__chain']:
            if type(value)==Modifier:
                nmod=nmod+1
                if details: self.inspectModifier(value,details)
        print '   ',nmod,'modifiers defined' 

    def inspect(self,name=None,onlyChosen=False,details=True):
        if name==None:
            print 'Inspecting the known eras',
            if onlyChosen: print ' (all active)'
            else: print '(all eras defined)'
        else:
            print 'Inspecting the '+name+' era',

        allEras=[]
        for key, value in self.__dict__.items():
            if type(value)==ModifierChain: allEras.append(key)

        for e in allEras:
            if name is not None and name==e:
                self.inspectEra(e,details)
            if name is None:
                if not onlyChosen or getattr(self,e).isChosen(): 
                    self.inspectEra(e,details)


eras=Eras()

allEras=['Run2_50ns',
         'Run2_25ns',
         'Run2_HI',
         'Run2_2016',
         'Run2_2016_trackingLowPU',
         'Run2_2017',
         'Run2_2017_trackingRun2',
         'Run2_2017_trackingPhase1PU70',
         'Run3',
         'Phase2C1',
         'Phase2C2']

for e in allEras:
    eObj=getattr(__import__('Configuration.Eras.Era_'+e+'_cff',globals(),locals(),[e],0),e)
    eras.addEra(e,eObj)

#eras.inspect()
