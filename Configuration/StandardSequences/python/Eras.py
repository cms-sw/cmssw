from __future__ import print_function
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
                 'Run2_2017_FastSim', #new modifier for Phase1 FastSim, skips the muon GEM sequence
                 'Run2_2017_trackingRun2',
                 'Run2_2017_trackingLowPU',
                 'Run2_2017_pp_on_XeXe',
                 'Run2_2017_ppRef',
                 'Run2_2018',
                 'Run2_2018_FastSim', #new modifier for Phase1 FastSim, skips the muon GEM sequence
                 'Run2_2018_pp_on_AA',
                 'Run2_2018_pp_on_AA_noHCALmitigation',
                 'Run2_2018_highBetaStar',
                 'Run3',
                 'Run3_pp_on_PbPb',
                 'Phase2',
                 'Phase2C4',
                 'Phase2C6',
                 'Phase2C8',
                 'Phase2C9',
                 'Phase2C10',
        ]

        internalUseMods = ['run2_common', 'run2_25ns_specific',
                           'run2_50ns_specific', 'run2_HI_specific',
                           'stage1L1Trigger', 'fastSim',
                           'peripheralPbPb', 'pA_2016',
                           'run2_HE_2017', 'stage2L1Trigger', 'stage2L1Trigger_2017', 'stage2L1Trigger_2018', 'stage2L1Trigger_2021',
                           'run2_HF_2017', 'run2_HCAL_2017', 'run2_HEPlan1_2017', 'run2_HB_2018','run2_HE_2018', 
                           'run3_HB', 'run3_common', 'run3_RPC',
                           'phase1Pixel', 'run3_GEM', 'run2_GEM_2017',
                           'run2_CSC_2018',
                           'phase2_common', 'phase2_tracker',
                           'phase2_hgcal', 'phase2_muon', 'phase2_timing', 'phase2_hgcalV9', 'phase2_hfnose', 'phase2_hgcalV10', 'phase2_hgcalV11',
                           'phase2_timing_layer', 'phase2_hcal',
                           'phase2_trigger',
                           'trackingLowPU', 'trackingPhase1', 'ctpps_2016', 'ctpps_2017', 'ctpps_2018', 'trackingPhase2PU140','highBetaStar_2018',
                           'tracker_apv_vfp30_2016', 'pf_badHcalMitigation', 'run2_miniAOD_80XLegacy','run2_miniAOD_94XFall17', 'run2_nanoAOD_92X',
                           'run2_nanoAOD_94XMiniAODv1', 'run2_nanoAOD_94XMiniAODv2', 'run2_nanoAOD_94X2016',
                           'run2_miniAOD_devel', 'run2_nanoAOD_102Xv1', 'run2_nanoAOD_106Xv1',
                           'hcalHardcodeConditions', 'hcalSkipPacker',
                           'run2_HLTconditions_2016','run2_HLTconditions_2017','run2_HLTconditions_2018',
                           'bParking']
        internalUseModChains = ['run2_2017_noTrackingModifier']

        self.pythonCfgLines = {}

        for e in allEras:
            eObj=getattr(__import__('Configuration.Eras.Era_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)
            self.pythonCfgLines[e] = 'from Configuration.Eras.Era_'+e+'_cff import '+e

        for e in internalUseMods:
            eObj=getattr(__import__('Configuration.Eras.Modifier_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)
            self.pythonCfgLines[e] = 'from Configuration.Eras.Modifier_'+e+'_cff import '+e

        for e in internalUseModChains:
            eObj=getattr(__import__('Configuration.Eras.ModifierChain_'+e+'_cff',globals(),locals(),[e],0),e)
            self.addEra(e,eObj)
            self.pythonCfgLines[e] = 'from Configuration.Eras.ModifierChain_'+e+'_cff import '+e


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
