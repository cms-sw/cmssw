from math import *

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event

import ROOT

##__________________________________________________________________||
class AlphaTAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(AlphaTAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.alphaTCalc = ROOT.heppy.AlphaT()

        self.usePt = hasattr(self.cfg_ana, 'usePt') and self.cfg_ana.usePt

    def process(self, event):
        self.readCollections( event.input )

        jets = getattr(event, self.cfg_ana.jets)

        if self.cfg_ana.jetSelectionFunc is not None:
            jets = [j for j in jets if self.cfg_ana.jetSelectionFunc(j)]

        alphaT, minDeltaHT, jetFlags = self.makeAlphaT(jets)

        setattr(event, self.cfg_ana.alphaT, alphaT)

        if self.cfg_ana.minDeltaHT is not None: setattr(event, self.cfg_ana.minDeltaHT, minDeltaHT)

        if self.cfg_ana.pseudoJetFlag is not None:
            for i, jet in enumerate(getattr(event, self.cfg_ana.jets)):
                pseudoJetFlag = jetFlags[i] if i < len(jetFlags) else -1
                setattr(jet, self.cfg_ana.pseudoJetFlag, pseudoJetFlag)

        if self.cfg_ana.inPseudoJet is not None:
            for i, jet in enumerate(getattr(event, self.cfg_ana.jets)):
                inPseudoJet = i < len(jetFlags)
                setattr(jet, self.cfg_ana.inPseudoJet, inPseudoJet)

        return True

    def makeAlphaT(self, jets):

        if len(jets) < 2: return -1, 0, [ ] # alphat, minDeltaHT, jetFlags
        
        jets = jets[:10] # use lead 10 jets

        px  = ROOT.std.vector('double')()
        py  = ROOT.std.vector('double')()

        for jet in jets:
            px.push_back(jet.px())
            py.push_back(jet.py())

        et  = ROOT.std.vector('double')()

        if self.usePt:
            for jet in jets: et.push_back(jet.pt())
        else:
            for jet in jets: et.push_back(jet.et())

        minDeltaHT = ROOT.Double(0.)
        jetFlags   = ROOT.std.vector('int')()

        alphaT =  self.alphaTCalc.getAlphaT(et, px, py, jetFlags, minDeltaHT)
        return alphaT, float(minDeltaHT), list(jetFlags)

##__________________________________________________________________||
