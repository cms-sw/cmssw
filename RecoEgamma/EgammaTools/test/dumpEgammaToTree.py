#!/usr/bin/env python
from array import array
import argparse
import sys
from DataFormats.FWLite import Events, Handle
import ROOT

def make_leaf_names(names,type_):
    """converts a list of names to a leaf string for a branch
    goes to the format [0]/{type_}:[1]:...[n] 
    also changes '-' to '_'
    """
    try:
        ret_val = '{}/{}'.format(names[0],type_)
        try: 
            return (ret_val+":"+':'.join(names[1:])).replace('-','_')
        except KeyError:
            return ret_val.replace('-','_')
    except KeyError:
        return ''
     
class EleTreeData:
    def __init__(self,tree_name):
        self.tree = ROOT.TTree(tree_name,'')
        self.initialised = False

    def _init_tree(self,ele):
        self.evt_data_names = ['runnr','lumisec','eventnr']
        self.ele_data_names = ['et','eta','phi','energy']
        self.ele_userfloat_names = [name for name in ele.userFloatNames()]
        self.ele_userint_names = [name for name in ele.userIntNames()]
        self.ele_id_names = [x.first for x in ele.electronIDs()]

        self.evt_data = array('I',[0]*len(self.evt_data_names))
        self.ele_data = array('f',[0.]*len(self.ele_data_names))
        self.ele_userfloat = array('f',[0.]*len(self.ele_userfloat_names))
        self.ele_userint = array('i',[0]*len(self.ele_userint_names))
        self.ele_id = array('i',[0]*len(self.ele_id_names))
        self.tree.Branch('evt',self.evt_data,make_leaf_names(self.evt_data_names,"i"))
        self.tree.Branch('ele',self.ele_data,make_leaf_names(self.ele_data_names,"F"))
        self.tree.Branch('eleFloats',self.ele_userfloat,make_leaf_names(self.ele_userfloat_names,"F"))
        self.tree.Branch('eleInts',self.ele_userint,make_leaf_names(self.ele_userint_names,"I"))
        self.tree.Branch('eleIds',self.ele_id,make_leaf_names(self.ele_id_names,"I"))
        self.initialised = True

    def fill(self,ele,event):
        if not self.initialised:
            self._init_tree(ele)

        self.evt_data[0] = event.eventAuxiliary().run()
        self.evt_data[1] = event.eventAuxiliary().luminosityBlock()
        self.evt_data[2] = event.eventAuxiliary().event()

        for indx,name in enumerate(self.ele_data_names):
            self.ele_data[indx] = getattr(ele,name)()
        for indx,name in enumerate(self.ele_userfloat_names):
            self.ele_userfloat[indx] = ele.userFloat(name)
        for indx,name in enumerate(self.ele_userint_names):
            self.ele_userint[indx] = ele.userInt(name)
        for indx,name in enumerate(self.ele_id_names):
            self.ele_id[indx] = int(ele.electronID(name))

        self.tree.Fill()

class PhoTreeData:
    def __init__(self,tree_name):
        self.tree = ROOT.TTree(tree_name,'')
        self.initialised = False

    def _init_tree(self,pho):
        self.evt_data_names = ['runnr','lumisec','eventnr']
        self.pho_data_names = ['et','eta','phi','energy']
        self.pho_userfloat_names = [name for name in pho.userFloatNames()]
        self.pho_userint_names = [name for name in pho.userIntNames()]
        self.pho_id_names = [x.first for x in pho.photonIDs()]

        self.evt_data = array('I',[0]*len(self.evt_data_names))
        self.pho_data = array('f',[0.]*len(self.pho_data_names))
        self.pho_userfloat = array('f',[0.]*len(self.pho_userfloat_names))
        self.pho_userint = array('i',[0]*len(self.pho_userint_names))
        self.pho_id = array('i',[0]*len(self.pho_id_names))
        self.tree.Branch('evt',self.evt_data,make_leaf_names(self.evt_data_names,"i"))
        self.tree.Branch('pho',self.pho_data,make_leaf_names(self.pho_data_names,"F"))
        self.tree.Branch('phoFloats',self.pho_userfloat,make_leaf_names(self.pho_userfloat_names,"F"))
        self.tree.Branch('phoInts',self.pho_userint,make_leaf_names(self.pho_userint_names,"I"))
        self.tree.Branch('phoIds',self.pho_id,make_leaf_names(self.pho_id_names,"I"))
        self.initialised = True

    def fill(self,pho,event):
        if not self.initialised:
            self._init_tree(pho)

        self.evt_data[0] = event.eventAuxiliary().run()
        self.evt_data[1] = event.eventAuxiliary().luminosityBlock()
        self.evt_data[2] = event.eventAuxiliary().event()

        for indx,name in enumerate(self.pho_data_names):
            self.pho_data[indx] = getattr(pho,name)()
        for indx,name in enumerate(self.pho_userfloat_names):
            self.pho_userfloat[indx] = pho.userFloat(name)
        for indx,name in enumerate(self.pho_userint_names):
            self.pho_userint[indx] = pho.userInt(name)
        for indx,name in enumerate(self.pho_id_names):
            self.pho_id[indx] = int(pho.photonID(name))

        self.tree.Fill()

oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

parser = argparse.ArgumentParser(description='prints E/gamma pat::Electrons/Photons us')
parser.add_argument('in_filename',help='input filename')
parser.add_argument('out_filename',help='output filename')
parser.add_argument('--min_pho_et','-p',default=20.,type=float,help='minimum photon et')
parser.add_argument('--min_ele_et','-e',default=20.,type=float,help='minimum electron et')
args = parser.parse_args()

eles, ele_label = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
phos, pho_label = Handle("std::vector<pat::Photon>"), "slimmedPhotons"


out_file = ROOT.TFile(args.out_filename,"RECREATE")
ele_tree = EleTreeData('eleTree')
pho_tree = PhoTreeData('phoTree')

events = Events(args.in_filename)
for event_nr,event in enumerate(events):
    
    event.getByLabel(pho_label,phos)
    
    for pho_nr,pho in enumerate(phos.product()):  
        if pho.et()>=args.min_pho_et: 
            pho_tree.fill(pho,event)

    event.getByLabel(ele_label,eles)
    for ele_nr,ele in enumerate(eles.product()):
        if ele.et()>=args.min_ele_et: 
            ele_tree.fill(ele,event)

out_file.Write()
