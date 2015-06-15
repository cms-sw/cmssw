from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average

from CMGTools.RootTools.statistics.TreeNumpy import TreeNumpy

from ROOT import TFile, TH1F

class NJetsAnalyzer(Analyzer):
    # class NJetsAnalyzer( GenParticleAnalyzer ):

    '''saves the NUP variable from the LHEEventProduct information.

    For the W+jets case:
    NUP = 5 : 0jets
    NUP = 6 : 1jet 
    ...

    In case of data, NUP = -1.
    In case of other MCs, the value is saved.
    '''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(NJetsAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

        # wpat = re.compile('(DY|W)\d?Jet.*') # match DY1Jet, DYJet, W1Jet, WJet, etc.
        # match = wpat.match(self.cfg_comp.name)
        # self.isWJets = not (match is None)

        # if self.isWJets:
        self.applyWeight = False
        if hasattr(self.cfg_comp, 'nevents'):
            assert(hasattr(self.cfg_comp, 'fractions'))
            assert(len(self.cfg_comp.nevents) == len(self.cfg_comp.fractions))
            self.ninc = self.cfg_comp.nevents[0]
            self.cfg_comp.nevents[0] = 0.
            self.ni = [frac * self.ninc for frac in self.cfg_comp.fractions]
            self.weighti = []
            for ninc, nexc in zip(self.ni, self.cfg_comp.nevents):
                self.weighti.append(ninc / (ninc + nexc))
            self.applyWeight = True

    def beginLoop(self, setup):
        super(NJetsAnalyzer, self).beginLoop(setup)
        self.averages.add('NUP', Average('NUP'))
        self.averages.add('NJets', Average('NJets'))
        self.averages.add('NJetWeight', Average('NJetWeight'))
        if self.cfg_comp.isMC:
            self.rootfile = TFile('/'.join([self.dirName,
                                            'NUP.root']),
                                  'recreate')
            self.nup = TH1F('nup', 'nup', 20, 0, 20)
            self.njets = TH1F('njets', 'njets', 10, 0, 10)
            self.tree = TreeNumpy('tree', 'test tree for NJetsAnalyzer')
            if self.cfg_ana.fillTree:
                self.tree.var('njets', int)
                self.tree.var('nup', int)
                self.tree.var('weight')

    def process(self, event):
        event.NUP = -1
        event.NJetWeight = 1

        if not self.cfg_comp.isMC:
            return True

        if not self.applyWeight:
            return True

        # JAN: FIXME: No HEP event product in W+Jets extensions, but we know NUP
        # from the file name
        if 'ext' in self.cfg_comp.name:
            event.NUP = int(self.cfg_comp.name[1]) + 5
        else:
            self.readCollections(event.input)
            event.NUP = self.mchandles['source'].product().hepeup().NUP

        # removing the 2 incoming partons, a boson,
        # and the 2 partons resulting from the decay of a boson
        njets = event.NUP - 5
        event.NJetWeight = self.weighti[njets]
        event.eventWeight *= event.NJetWeight

        if self.cfg_ana.fillTree:
            self.tree.reset()
            self.tree.fill('njets', njets)
            self.tree.fill('nup', event.NUP)
            self.tree.fill('weight', event.NJetWeight)
            self.tree.tree.Fill()

        self.averages['NUP'].add(event.NUP)
        self.averages['NJets'].add(njets)
        self.averages['NJetWeight'].add(event.NJetWeight)
        self.nup.Fill(event.NUP)
        self.njets.Fill(njets)

        if self.cfg_ana.verbose:
            print 'NUP, njets, weight', event.NUP, njets, event.NJetWeight
        return True

    def declareHandles(self):
        '''Reads LHEEventsProduct.'''
        super(NJetsAnalyzer, self).declareHandles()
        self.mchandles['source'] = AutoHandle(
            'source',
            'LHEEventProduct'
        )

    def write(self, setup):
        super(NJetsAnalyzer, self).write(setup)
        if self.cfg_comp.isMC:
            self.rootfile.Write()
            self.rootfile.Close()
