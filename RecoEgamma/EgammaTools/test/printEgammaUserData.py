#!/usr/bin/env python

def convert_to_str(vec_str):
    output = ""
    for entry in vec_str:
        if output != "": output+="\n  "
        output+=entry
    return output

def convertpair_to_str(vec_str):
    output = ""
    for entry in vec_str:
        if output != "": output+="\n  "
        output+=entry.first
    return output

def print_ele_user_data(ele):
    print "ele userfloats:"
    print "  "+convert_to_str(ele.userFloatNames())
    print "ele userints:"
    print "  "+convert_to_str(ele.userIntNames())
    print "ele IDs:"
    print "  "+convertpair_to_str(ele.electronIDs())

def print_pho_user_data(pho):
    print "pho userfloats:"
    print "  "+convert_to_str(pho.userFloatNames())
    print "pho userints:"
    print "  "+convert_to_str(pho.userIntNames())
    print "pho IDs:"
    print "  "+convertpair_to_str(pho.photonIDs())


import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Events, Handle

import argparse
parser = argparse.ArgumentParser(description='prints E/gamma pat::Electrons/Photons user data')
parser.add_argument('filename',help='input filename')
args = parser.parse_args()

eles, ele_label = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
phos, pho_label = Handle("std::vector<pat::Photon>"), "slimmedPhotons"


min_pho_et = 20
min_ele_et = 20

done_ele = False
done_pho = False

events = Events(args.filename)
for event_nr,event in enumerate(events):
    if done_ele and done_pho: break
    
    if not done_pho:
        event.getByLabel(pho_label,phos)
    
        for pho_nr,pho in enumerate(phos.product()):  
            if pho.et()<min_pho_et: continue
            else:
                print_pho_user_data(pho)
                done_pho = True
    if not done_ele:
        event.getByLabel(ele_label,eles)
    
        for ele_nr,ele in enumerate(eles.product()):
            if ele.et()<min_ele_et: continue
            else:
                print_ele_user_data(ele)
                done_ele = True
