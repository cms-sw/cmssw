import re

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average

from ROOT import TFile, TH1F

class WNJetsAnalyzer( Analyzer ):
#class WNJetsAnalyzer( GenParticleAnalyzer ):
    '''saves the NUP variable from the LHEEventProduct information.
    
    For the W+jets case:
    NUP = 5 : 0jets
    NUP = 6 : 1jet 
    ...
    
    In case of data, NUP = -1.
    In case of other MCs, the value is saved.
    '''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(WNJetsAnalyzer,self).__init__(cfg_ana, cfg_comp, looperName)

        wpat = re.compile('W\d?Jet.*')
        match = wpat.match(self.cfg_comp.name)
        self.isWJets = not (match is None)

        if self.isWJets:
            self.ninc = self.cfg_ana.nevents[0]
            self.cfg_ana.nevents[0] = 0.
            self.ni = [frac*self.ninc for frac in self.cfg_ana.fractions]
            assert(len(self.cfg_ana.nevents)==len(self.cfg_ana.fractions))
            self.weighti = []
            for ninc, nexc in zip(self.ni, self.cfg_ana.nevents ):
                self.weighti.append( ninc/(ninc+nexc) )


        
    def beginLoop(self, setup):
        super(WNJetsAnalyzer,self).beginLoop(setup)        
        self.averages.add('NUP', Average('NUP') )
        self.averages.add('NJets', Average('NJets') )
        self.averages.add('WJetWeight', Average('WJetWeight') )
        if self.cfg_comp.isMC:
            self.rootfile = TFile('/'.join([self.dirName,
                                            'NUP.root']),
                                  'recreate')
            self.nup = TH1F('nup', 'nup', 20,0,20)
            self.njets = TH1F('njets', 'njets', 10,0,10)
        
        
    def process(self, event):
        event.NUP = -1
        event.WJetWeight = 1
        
        if not self.cfg_comp.isMC:
            return True

        if not self.isWJets:
            return True
        
        #        try:
        self.readCollections( event.input )
        event.NUP = self.mchandles['source'].product().hepeup().NUP
        # except :
        #    return True
        # assert(event.NUP>0)

        # removing the 2 incoming partons, a boson,
        # and the 2 partons resulting from the decay of a boson
        njets = event.NUP-5
        event.WJetWeight = self.weighti[njets]
        event.eventWeight *= event.WJetWeight

        self.averages['NUP'].add( event.NUP )
        self.averages['NJets'].add( njets )
        self.averages['WJetWeight'].add( event.WJetWeight )
        self.nup.Fill(event.NUP)
        self.njets.Fill(njets)

        
        if self.cfg_ana.verbose:
            print 'NUP, njets, weight',event.NUP, njets, event.WJetWeight
        return True
    

    def declareHandles(self):
        '''Reads LHEEventsProduct.'''
        super(WNJetsAnalyzer, self).declareHandles()
        self.mchandles['source'] =  AutoHandle(
            'source',
            'LHEEventProduct'
            )
        
    def write(self):
        super(WNJetsAnalyzer, self).write()
        if self.cfg_comp.isMC:
            self.rootfile.Write()
            self.rootfile.Close()
