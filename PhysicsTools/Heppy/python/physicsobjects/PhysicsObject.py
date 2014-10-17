import copy
from PhysicsTools.Heppy.physicsobjects.Particle import Particle

#COLIN should make a module for lorentz vectors (and conversions)
#instanciating template
from ROOT import Math
PtEtaPhiE4DLV = Math.PtEtaPhiE4D(float)
PtEtaPhiM4DLV = Math.PtEtaPhiM4D(float)


class PhysicsObject(Particle):
    '''Extends the cmg::PhysicsObject functionalities.'''

    def __init__(self, physObj):
        self.physObj = physObj
        super(PhysicsObject, self).__init__()

    def __copy__(self):
        '''Very dirty trick, the physObj is deepcopied...'''
        # print 'call copy', self
        physObj = copy.deepcopy( self.physObj )
        newone = type(self)(physObj)
        newone.__dict__.update(self.__dict__)
        newone.physObj = physObj
        return newone        

    def scaleEnergy( self, scale ):
        p4 = self.physObj.p4()
        p4 *= scale 
        self.physObj.setP4( p4 )  
##         p4 = self.physObj.polarP4()
##         sp4 = PtEtaPhiE4DLV(
##             p4.Pt()*scale,
##             p4.Eta(),
##             p4.Phi(),
##             p4.E()
##             )
##         sp4.SetE( sp4.E()*scale )
##         ptepmsp4 = PtEtaPhiM4DLV(
##             sp4.Pt(),
##             sp4.Eta(),
##             sp4.Phi(),
##             sp4.M()
##             )
##         self.physObj.setP4( p4.__class__(
##             sp4.Pt(),
##             sp4.Eta(),
##             sp4.Phi(),
##             sp4.M()
##             ) )
        
        
    def __getattr__(self,name):
        '''all accessors  from cmg::DiTau are transferred to this class.'''
        return getattr(self.physObj, name)

