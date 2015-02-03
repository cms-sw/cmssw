from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
import ROOT
import math

class IsoTrack( PhysicsObject ):
    
    def __init__(self, isoTrack):
        self.isoTrack = isoTrack
        super(IsoTrack, self).__init__(isoTrack)

    def absIso(self, dummy1, dummy2):
        '''Just making the tau behave as a lepton.'''
        return -1

    def relIso(self, dummy1, dummy2):
        '''Just making the tau behave as a lepton.'''
        return -1

    def __str__(self):
        lep = super(IsoTrack, self).__str__()
        return lep
        #spec = '\t\tTau: decay = {decMode:<15}, eOverP = {eOverP:4.2f}, isoMVALoose = {isoMVALoose}'.format(
        #    decMode = tauDecayModes.intToName( self.decayMode() ),
        #    eOverP = self.calcEOverP(),
        #    isoMVALoose = self.tauID('byLooseIsoMVA')
        #    )
        #return '\n'.join([lep, spec])


