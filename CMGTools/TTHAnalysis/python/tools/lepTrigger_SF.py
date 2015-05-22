#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
import re,os

class LepTriggerSF_Event:
    def __init__(self,file="NotreDame_lepton_SFs.root"):
        tfile = ROOT.TFile("%s/src/CMGTools/TTHAnalysis/data/%s" % (os.environ['CMSSW_BASE'],file))
        self.hmm = tfile.Get("TwoMuonTriggerSF").Clone("LepTriggerSF_TwoMuonTriggerSF")
        self.hee = tfile.Get("TwoEleTriggerSF").Clone("LepTriggerSF_TwoEleTriggerSF")
        self.hme = tfile.Get("MuonEleTriggerSF").Clone("LepTriggerSF_MuonEleTriggerSF")
        self.hmm.SetDirectory(None)
        self.hee.SetDirectory(None)
        self.hme.SetDirectory(None)
        tfile.Close()
    def __call__(self,event):
        leps = Collection(event,"LepGood","nLepGood",4)
        if len(leps) < 2: return 1.0
        id1,id2 = abs(leps[0].pdgId), abs(leps[1].pdgId)
        if id1 == id2:
            h = self.hmm if id1 == 13 else self.hee
            return h.GetBinContent(h.FindBin(min(2.09,abs(leps[0].eta)), min(2.49,abs(leps[1].eta))))
        else:
            (mueta,eleta) = (leps[0].eta,leps[1].eta) if id1 == 13 else (leps[1].eta,leps[0].eta)
            return self.hme.GetBinContent(self.hme.FindBin(min(2.09,abs(mueta)), min(2.49,abs(eleta))))

 

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = LepTriggerSF_Event()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 100)
