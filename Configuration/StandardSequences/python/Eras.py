import FWCore.ParameterSet.Config as cms
from  FWCore.ParameterSet.Config import ModifierChain,Modifier

class Eras (object):
    """
    Dummy container for all the cms.Modifier instances that config fragments
    can use to selectively configure depending on what scenario is active.
    """
    def __init__(self):
        allEras=['Run1_pA',
                 'Run1_peripheralPbPb',
                 'Run2_50ns',
                 'Run2_50ns_HIPM',
                 'Run2_25ns',
                 'Run2_25ns_HIPM',
                 'Run2_25ns_peripheralPbPb',
                 'Run2_HI',
                 'Run2_2016',
                 'Run2_2016_HIPM',
                 'Run2_2016_trackingLowPU',
                 'Run2_2016_pA',
                 'Run2_2017',
                 'Run2_2017_trackingRun2',
                 'Run2_2017_trackingPhase1QuadProp',
                 'Run2_2017_trackingLowPU',
                 'Run2_2018',
                 'Run3',
                 'Phase2C1',
                 'Phase2C2',
                 'Phase2C1_timing',
                 'Phase2C2_timing',
                 'Phase2C2_timing_layer']

        internalUseMods = ['run2_common', 'run2_25ns_specific',
                           'run2_50ns_specific', 'run2_HI_specific',
                           'stage1L1Trigger', 'fastSim',
                           'peripheralPbPb', 'pA_2016',
                           'run2_HE_2017', 'stage2L1Trigger',
                           'run2_HF_2017', 'run2_HCAL_2017', 'run2_HEPlan1_2017',
                           'run3_HB',
                           'phase1Pixel', 'run3_GEM', 'run2_GEM_2017',
                           'phase2_common', 'phase2_tracker',
                           'phase2_hgcal', 'phase2_muon', 'phase2_timing',
                           'phase2_timing_layer','phase2_hcal',
                           'trackingLowPU', 'trackingPhase1', 'trackingPhase1QuadProp', 'ctpps_2016', 'trackingPhase2PU140',
                           'tracker_apv_vfp30_2016']
        internalUseModChains = ['run2_2017_noTrackingModifier']


        for e in allEras:
            eObj=getattr(__import__('Configuration.Eras.Era_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)

        for e in internalUseMods:
            eObj=getattr(__import__('Configuration.Eras.Modifier_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)

        for e in internalUseModChains:
            eObj=getattr(__import__('Configuration.Eras.ModifierChain_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)


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


#eras.inspect()
