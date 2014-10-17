from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.Heppy.physicsobjects.TauDecayModes import tauDecayModes
import math

cutsElectronMVA3Medium = [0.933,0.921,0.944,0.945,0.918,0.941,0.981,0.943,0.956,0.947,0.951,0.95,0.897,0.958,0.955,0.942]

class Tau( Lepton ):
    
    def __init__(self, tau):
        self.tau = tau
        super(Tau, self).__init__(tau)
        self.eOverP = None

    # JAN FIXME - check what's available in miniAOD and if we need this at all
    def calcEOverP(self):
        if self.eOverP is not None:
            return self.eOverP
        self.leadChargedEnergy = self.tau.leadChargedHadrEcalEnergy() \
                                 + self.tau.leadChargedHadrHcalEnergy()
        # self.leadChargedMomentum = self.tau.leadChargedHadrPt() / math.sin(self.tau.theta())
        self.leadChargedMomentum = self.tau.leadPFChargedHadrCand().energy()
        self.eOverP = self.leadChargedEnergy / self.leadChargedMomentum
        return self.eOverP         

    def relIso(self, dummy1, dummy2):
        '''Just making the tau behave as a lepton.'''
        return -1

    def mvaId(self):
        '''For a transparent treatment of electrons, muons and taus. Returns -99'''
        return -99

    def dxy(self, vertex=None):
        if vertex is None:
            vertex = self.associatedVertex
        vtx = self.vertex() # FIXME 
        p4 = self.p4()
        return ( - (vtx.x()-vertex.position().x()) *  p4.y()
                 + (vtx.y()-vertex.position().y()) *  p4.x() ) /  p4.pt()

    def dz(self, vertex=None):
        if vertex is None:
            vertex = self.associatedVertex
        vtx = self.vertex()  # FIXME 
        p4 = self.p4()
        return  (vtx.z()-vertex.position().z()) - ((vtx.x()-vertex.position().x())*p4.x()+(vtx.y()-vertex.position().y())*p4.y())/ p4.pt() *  p4.z()/ p4.pt()
    
    def zImpact(self, vertex=None):
        '''z impact at ECAL surface'''
        if vertex is None:
            vertex = self.associatedVertex
        return vertex.z() + 130./math.tan(self.theta())

    def __str__(self):
        lep = super(Tau, self).__str__()
        return lep
        #spec = '\t\tTau: decay = {decMode:<15}, eOverP = {eOverP:4.2f}, isoMVALoose = {isoMVALoose}'.format(
        #    decMode = tauDecayModes.intToName( self.decayMode() ),
        #    eOverP = self.calcEOverP(),
        #    isoMVALoose = self.tauID('byLooseIsoMVA')
        #    )
        #return '\n'.join([lep, spec])

    def electronMVA3Medium(self):
        '''Custom electron MVA 3 medium WP used for H->tau tau'''
        icat = int(round(self.tauID('againstElectronMVA3category')))
        if icat < 0:
            return False
        elif icat > 15:
            return True

        rawMVA = self.tauID('againstElectronMVA3raw') 
        return rawMVA > cutsElectronMVA3Medium[icat]


def isTau(leg):
    '''Duck-typing a tau'''
    try:
        # Taken from BaseTau to work for both PFTaus and PAT Taus
        # (can maybe find a less expensive method)
        leg.leadTrack()
    except AttributeError:
        return False
    return True

