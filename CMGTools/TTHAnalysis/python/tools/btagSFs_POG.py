#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
import re,os

class BTag_SFb:
    def __init__(self,filename):
        self._SFb = {}
        self._SFbErrs = {}
        self._ptbins = []
        file = open(filename, "r")
        ptminLine = file.readline()
        ptmaxLine = file.readline()
        if not re.match(r"float\s*ptmin\[\]\s*=\s*\{\s*(\d+\,\s*)+\d+\s*\}\s*;\s*\n", ptminLine):
            raise RuntimeError, "malformed ptmin line %s" % ptminLine;
        if not re.match(r"float\s*ptmax\[\]\s*=\s*\{\s*(\d+\,\s*)+\d+\s*\}\s*;\s*\n", ptmaxLine):
            raise RuntimeError, "malformed ptmax line %s" % ptmaxLine;
        m = re.match(r"float\s*ptmin\[\]\s*=\s*\{(\s*(\d+\,\s*)+\d+\s*)\}\s*;\s*\n", ptminLine)
        ptmins = [ int(x.strip()) for x in m.group(1).split(",") ]
        m = re.match(r"float\s*ptmax\[\]\s*=\s*\{(\s*(\d+\,\s*)+\d+\s*)\}\s*;\s*\n", ptmaxLine)
        ptmaxs = [ int(x.strip()) for x in m.group(1).split(",") ]
        for i,pt in enumerate(ptmaxs[:-1]):
            if ptmins[i+1] != pt: raise RuntimeError, "ptMin and ptMax don't match!!"
        self._ptbins = ptmaxs
        tagger = None
        for line in file:
            m = re.match(r"\s*Tagger: (\S+[LMT]) within.*", line)
            if m:
                tagger = m.group(1)
                sfbline = file.next()
                m = re.match(r"\s*SFb = (.*);", sfbline)
                self._SFb[tagger] = eval("lambda x : "+m.group(1))
                self._SFbErrs[tagger] = []
                #print "Found tagger",tagger,": SFb = ",m.group(1)
            if re.match("\s*SFb_error\[\]\s*=\s*\{",line):
                for ib,b in enumerate(self._ptbins):
                    errline = file.next()
                    m =  re.match(r"\s*(\d+\.\d+)\s*(,|\}).*", errline)
                    if not m: raise RuntimeError, "Missing uncertainty for pt bin %s for tagger %s" % (b,tagger)
                    self._SFbErrs[tagger].append(float(m.group(1)))
                    if (b == self._ptbins[-1]) != ("}" == m.group(2)):
                        if "CSVSLV1" in tagger and "}" == m.group(2):
                            for b2 in self._ptbins[ib+1:]:
                                self._SFbErrs[tagger].append(2*float(m.group(1)))
                            break
                        else:
                            raise RuntimeError, "Mismatching uncertainties for tagger %s at line %s" % (tagger,errline)
    def __call__(self,tagger,jet,syst=0):
        ret = self._SFb[tagger](jet.pt)
        if len(self._SFbErrs[tagger]) != len(self._ptbins):
            raise RuntimeError, "Mismatching uncertainties for tagger %s" % (tagger,)
        if syst != 0:
           jpt = jet.pt
           for (pt,err) in zip(self._ptbins,self._SFbErrs[tagger]):
                if pt <= jpt: 
                    ret += syst*err
                    break
        return ret
    
class BTag_SFLight:
    def __init__(self,filename):
        self._SFLight = {}
        file = open(filename, "r")
        tagline = re.compile(r'if\s*\(\s*Atagger\s*==\s*"(\S+)"\s*&&\s*sEtamin\s*==\s*"(\S+)"\s*&&\s*sEtamax\s*==\s*"(\S+)"\s*\)');
        sflline = re.compile(r'new\s+TF1\(\s*"(\S+)"\s*,\s*"(.*)"\s*,\s*(\d+\.?\d*)\s*,\s*ptmax\s*\)');
        tagger = None
        for line in file:
            tagm = re.search(tagline, line)
            if tagm:
                (tagger,etamin,etamax) = (tagm.group(1),float(tagm.group(2)),float(tagm.group(3)))
                if tagger not in self._SFLight: 
                    self._SFLight[tagger] = []
                self._SFLight[tagger].append([etamin,etamax,{}])
                continue
            sfm = re.search(sflline, line)
            if sfm:
                self._SFLight[tagger][-1][2][sfm.group(1)] = eval("lambda x: "+sfm.group(2));
    def __call__(self,tagger,jet,syst=0):
        what = {0:'SFlight', -1:'SFlightMin', +1:'SFlightMax'}[syst]
        ae = abs(jet.eta)
        for etamin, etamax, sflights in self._SFLight[tagger]:
            if etamin <= ae and ae <= etamax: 
                return sflights[what](jet.pt)
        return 1.0 

class BTagSFEvent:
    def __init__(self,sfb,sflight,effhist,WPs=[('CSVM',0.679), ('CSVL',0.244)]):
        self._sfb = sfb
        self._sflight = sflight
        self._efffile = ROOT.TFile.Open(effhist)
        self._effhists = {}
        self._WPs = WPs
        for T,C in self._WPs:
            for (l,f) in (('b',5), ('c',4), ('l',0)):
                self._effhists[(T,f)] = self._efffile.Get("%s_eff_%s" % (l,T))
    def _mceff(self,tagger,j):
        mcFlav = abs(j.mcFlavour) if j.mcFlavour in [4,5,-4,-5] else 0
        hist = self._effhists[(tagger,mcFlav)]
        xbin = hist.GetXaxis().FindBin(max(min(149.9,j.pt),25.1))
        ybin = hist.GetYaxis().FindBin(max(min(4.9,abs(j.eta)),0.1))
        #print "eff %s for pt %.1f, eta %.1f --> %.3f" % (j.pt,
        return hist.GetBinContent(xbin,ybin)
    def __call__(self,event,debug=False,systL=0,systB=0):
        jets = Collection(event,"Jet","nJet25",8)
        num, den = 1.0, 1.0
        for j in jets:
            tagged = False
            mcFlav = abs(j.mcFlavour) if j.mcFlavour in [4,5,-4,-5] else 0
            mySystB = systB if mcFlav != 4 else 2*systB
            for iw,(T,C) in enumerate(self._WPs):
                if j.btagCSV > C: 
                    tagged = True
                    eps = self._mceff(T,j)
                    sf  = self._sfb(T,j,mySystB) if mcFlav > 0 else self._sflight(T,j,systL)
                    if iw > 0:
                        TT,TC = self._WPs[iw-1]
                        epsT = self._mceff(TT,j)
                        sfT  = self._sfb(TT,j,mySystB) if mcFlav > 0 else self._sflight(TT,j,systL)
                        num *= max(0, sf*eps - sfT*epsT)
                        den *= max(0, eps - epsT)
                        if debug:
                            print "    jet pt %5.1f eta %+4.2f btag %4.3f mcFlavour %d --> pass %s (eff %.3f, sf %.3f) but fail %s (eff %.3f, sf %.3f) --> event contrib: %.4f" % (j.pt, j.eta, min(1.,max(0.,j.btagCSV)), j.mcFlavour, T, eps, sf, TT, epsT, sfT, max(0, sf*eps - sfT*epsT)/(eps - epsT) if eps-epsT > 0 else 1) 
                    else:
                        if debug:
                            print "    jet pt %5.1f eta %+4.2f btag %4.3f mcFlavour %d --> pass %s (eff %.3f, sf %.3f) --> event contrib: %.4f" % (j.pt, j.eta, min(1.,max(0.,j.btagCSV)), j.mcFlavour, T, eps, sf, sf) 
                        num *= sf*eps
                        den *= eps
                    break
            if not tagged:
                T,C = self._WPs[-1]
                eps = self._mceff(T,j)
                sf  = self._sfb(T,j,mySystB) if mcFlav > 0 else self._sflight(T,j,systL)
                if debug:
                    print "    jet pt %5.1f eta %+4.2f btag %4.3f mcFlavour %d --> fail %s (eff %.3f, sf %.3f) --> event contrib: %.4f" % (j.pt, j.eta, min(1.,max(0.,j.btagCSV)), j.mcFlavour, T, eps, sf, max(0, 1-sf*eps)/max(0, 1-eps) if eps < 1 else 1)
                num *= max(0, 1-sf*eps)
                den *= max(0, 1-eps)
        return num/den if den != 0 else 1;

class BTagSFEventErrs:
    def __init__(self,sfev):
        self.sfev = sfev
    def listBranches(self):
        return [ 'btag', 'btag_bUp', 'btag_bDown', 'btag_lUp', 'btag_lDown' ]
    def __call__(self,event):
        return {
            'btag':       self.sfev(event),
            'btag_bUp':   self.sfev(event,systB=+1),
            'btag_bDown': self.sfev(event,systB=-1),
            'btag_lUp':   self.sfev(event,systL=+1),
            'btag_lDown': self.sfev(event,systL=-1),
        }

bTagSFEvent3WP = BTagSFEvent(
    #BTag_SFb("%s/src/CMGTools/TTHAnalysis/data/btag/SFb-pt_payload_Moriond13.txt" % os.environ['CMSSW_BASE']),
    #BTag_SFLight("%s/src/CMGTools/TTHAnalysis/data/btag/SFLightFuncs_Moriond2013_ABCD.txt" % os.environ['CMSSW_BASE']),
    BTag_SFb("%s/src/CMGTools/TTHAnalysis/data/btag/SFb-pt_WITHttbar_payload_EPS13.txt" % os.environ['CMSSW_BASE']),
    BTag_SFLight("%s/src/CMGTools/TTHAnalysis/data/btag/SFlightFuncs_EPS2013.C.txt" % os.environ['CMSSW_BASE']),
    "%s/src/CMGTools/TTHAnalysis/data/btag/bTagEff_TTMC.root" % os.environ['CMSSW_BASE'],
    WPs=[('CSVT',0.898), ('CSVM',0.679), ('CSVL',0.244) ]
)
bTagSFEvent3WPErrs = BTagSFEventErrs(bTagSFEvent3WP)

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerTTH")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name, sfe1,sfe2,sfe3):
            Module.__init__(self,name,None)
            self.sfe1 = sfe1
            self.sfe2 = sfe2
            self.sfe3 = sfe3
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: jets %d" % (ev.run, ev.lumi, ev.evt, ev.nJet25)
            jets = Collection(ev,"Jet","nJet25",8)
            overall1 = self.sfe1(ev)
            overall2 = self.sfe2(ev)
            overall3 = self.sfe3(ev)
            print "overall %d %d %.4f %.4f %.4f" % (ev.nBJetLoose25,ev.nBJetMedium25,overall1,overall2,overall3)
    class Tester2(Module):
        def __init__(self, name, sf):
            Module.__init__(self,name,None)
            self.sf = sf
        def analyze(self,event):
            nominal  = self.sf(event)
            bup, bdn = self.sf(event,systB=+1), self.sf(event,systB=-1)
            lup, ldn = self.sf(event,systL=+1), self.sf(event,systL=-1)
            print "%d %d   %.3f   %.3f %.3f  %.3f %.3f" % (event.nBJetLoose25, event.nBJetMedium25, nominal, bdn, bup, ldn, lup)
    bTagSFEvent1WP = BTagSFEvent(
        BTag_SFb("%s/src/CMGTools/TTHAnalysis/data/btag/SFb-pt_payload_Moriond13.txt" % os.environ['CMSSW_BASE']),
        BTag_SFLight("%s/src/CMGTools/TTHAnalysis/data/btag/SFLightFuncs_Moriond2013_ABCD.txt" % os.environ['CMSSW_BASE']),
        "%s/src/CMGTools/TTHAnalysis/data/btag/bTagEff_TTMC.root" % os.environ['CMSSW_BASE'],
        WPs=[('CSVL',0.244)]
    )
    bTagSFEvent2WP = BTagSFEvent(
        BTag_SFb("%s/src/CMGTools/TTHAnalysis/data/btag/SFb-pt_payload_Moriond13.txt" % os.environ['CMSSW_BASE']),
        BTag_SFLight("%s/src/CMGTools/TTHAnalysis/data/btag/SFLightFuncs_Moriond2013_ABCD.txt" % os.environ['CMSSW_BASE']),
        "%s/src/CMGTools/TTHAnalysis/data/btag/bTagEff_TTMC.root" % os.environ['CMSSW_BASE'],
        WPs=[('CSVM',0.679), ('CSVL',0.244)]
    )
    #el = EventLoop([ Tester("tester", bTagSFEvent1WP, bTagSFEvent2WP, bTagSFEvent3WP) ])
    el = EventLoop([ Tester2("tester", bTagSFEvent3WP) ])
    el.loop([tree], maxEvents = 1000)
