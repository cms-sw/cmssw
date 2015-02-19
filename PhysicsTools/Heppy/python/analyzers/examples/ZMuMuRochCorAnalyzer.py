import copy

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.AutoHandle import AutoHandle

from PhysicsTools.Heppy.physicsobjects.RochesterCorrections import rochcor
from PhysicsTools.Heppy.physicsobjects.Particle import Particle
from PhysicsTools.Heppy.physicsobjects.Muon import Muon

from ROOT import gSystem
gSystem.Load("libDataFormatsRecoCandidate.so")
from ROOT import reco

class DiMuon( reco.LeafCandidate ):
    """ Simple DiMuon object
    
    using the LeafCandidate (~simple particle ) dataformat from CMSSW as a base.
    this class is made in such a way that it behaves almost exactly
    as physicsobjects.DiObjects.DiMuon. 
    """

    def __init__(self, l1, l2, diLepton):
        '''l1 and l2 are the legs, possibly recalibrated.
        diLepton is the original diLepton, used only in the met function'''
        self.diLepton = diLepton
        self.l1 = l1
        self.l2 = l2
        self.sumpt = l1.pt() + l2.pt()
        super(DiMuon, self).__init__(0, l1.p4()+l2.p4())

    def __str__(self):
        return 'DiMuon: mass={mass:5.2f}, sumpt={sumpt:5.2f}, pt={pt:5.2f}'.format(
            mass = self.mass(),
            sumpt = self.sumpt,
            pt = self.pt()
            )

    def met(self):
        '''this is going to be needed to compute VBF related quantities.
        just giving the met associated to the original di-lepton
        '''
        return self.diLepton.met()
    
    def leg1(self):
        return self.l1

    def leg2(self):
        return self.l2


class ZMuMuRochCorAnalyzer( Analyzer ):

    def process(self, iEvent, event):
        
        def correctDiLepton( diLepton ):
            '''Corrects a di-lepton.

            This function is defined within the process function to have
            access to the variables available there, namely event.run.
            The goal of this function is to be able to call it with map,
            to get very compact code.
            '''
            p4_1 = rochcor.corrected_p4( diLepton.leg1(), event.run ) 
            p4_2 = rochcor.corrected_p4( diLepton.leg2(), event.run )
            # l1 = copy.copy( diLepton.leg1() )
            # l2 = copy.copy( diLepton.leg2() )
            l1 = diLepton.leg1()
            l2 = diLepton.leg2()
            l1.setP4(p4_1)
            l2.setP4(p4_2)    
            diLeptonCor = DiMuon( l1, l2, diLepton)  
            return diLeptonCor

        event.diLeptonRaw = copy.copy(event.diLepton)
        event.diLepton = correctDiLepton( event.diLeptonRaw )
