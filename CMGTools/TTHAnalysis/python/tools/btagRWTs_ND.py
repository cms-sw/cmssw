#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
import re,os

class BTag_RWT_One:
    def __init__(self,rootfile,suffix,ptbins,etabins):
        self._etabins = etabins[1:]
        self._ptbins  = ptbins[1:]
        self._nptbins  = len(self._ptbins)
        self._netabins = len(self._etabins)
        self._hists = {}
        for ip in xrange(self._nptbins):
            for ie in xrange(self._netabins):
                histo = rootfile.Get("csv_ratio_Pt%d_Eta%d_%s" % (ip,ie,suffix))
                if histo:
                    histo = histo.Clone(); histo.SetDirectory(None)
                    self._hists[ip*self._netabins+ie] = histo
                else:
                    #rootfile.ls()
                    raise RuntimeError, "Error: missing 'csv_ratio_Pt%d_Eta%d_%s' in %s" % (ip,ie,suffix,rootfile.GetName())
    def findBin1D(self,var,list): # return the index of the first element in list that does not exceed var, or len(list)-1
        i = 1; n = len(list)
        while i < n and var > list[i-1]:
            i+=1
        return i-1
    def findBin2D(self,jet):
        ie = self.findBin1D(abs(jet.eta), self._etabins) 
        ip = self.findBin1D(abs(jet.pt),  self._ptbins) 
        return ip*self._netabins+ie
    def __call__(self,jet):
        csv = jet.btagCSV
        hist = self._hists[self.findBin2D(jet)]
        b = min(max(hist.GetXaxis().FindBin(jet.btagCSV), 1), hist.GetNbinsX())
        return hist.GetBinContent(b)
    
class BTag_RWT_Event:
    def __init__(self,syst=""):
        # heavy flavour
        filehf = ROOT.TFile("%s/src/CMGTools/TTHAnalysis/data/btag/csv_rwt_hf_IT.root" % os.environ['CMSSW_BASE']) 
        suffixhf = "final" if syst == "" else "final_%s" % syst
        self._whf = BTag_RWT_One(filehf, suffixhf, ptbins=(0,40,60,100,160,1000), etabins=(0,2.4))
        filehf.Close()
        # light flavour
        filelf = ROOT.TFile("%s/src/CMGTools/TTHAnalysis/data/btag/csv_rwt_lf_IT.root" % os.environ['CMSSW_BASE']) 
        suffixlf = suffixhf.replace("_LF","_HF")
        self._wlf = BTag_RWT_One(filelf, suffixlf, ptbins=(0,40,60,1000), etabins=(0,0.8,1.6,2.4))
        filelf.Close()
    def wjet(self,jet):
        w = self._whf if jet.mcFlavour in [4,5,-4,-5] else self._wlf
        return w(jet)
    def __call__(self,event,debug=False):
        jets = Collection(event,"Jet","nJet25",8)
        ret = 1.0
        for j in jets:
            wj = self.wjet(j)
            if debug:
                print "    jet pt %5.1f eta %+4.2f btag %4.3f mcFlavour %d ---> weight %.3f" % (j.pt, j.eta, j.btagCSV, j.mcFlavour, wj)
            if wj: ret *= wj
        return ret 

class BTag_RWT_EventErrs:
    def __init__(self):
        self._weights = { "btagRwt" : BTag_RWT_Event() }
        for K in 'JES', 'LF', 'Stats1', 'Stats2':
            self._weights["btagRwt_"+K+"Up"]   = BTag_RWT_Event(K+"Up")
            self._weights["btagRwt_"+K+"Down"] = BTag_RWT_Event(K+"Down")
    def listBranches(self):
        return self._weights.keys()
    def __call__(self,event):
        return dict( [ (k,w(event)) for (k,w) in self._weights.iteritems() ] )

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = BTag_RWT_Event()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: jets %d" % (ev.run, ev.lumi, ev.evt, ev.nJet25)
            print "\tnb %d weight %.3f" % (ev.nBJetLoose25, self.sf(ev,debug=True))
            #for k,v in self.sfe(ev).iteritems():
            #    print "\t%-20s: %5.3f" % (k,v)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 10000)
