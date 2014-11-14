from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

class GenParticle( PhysicsObject ):
    def __str__(self):
        base = super(GenParticle, self).__str__()
        theStr = '{base}, status = {status:>2}'.format(base=base, status=self.status())
        return theStr


class GenLepton( GenParticle ):
    def sip3D(self):
        '''Just to make generic code work on GenParticles'''
        return 0
    def relIso(self, dummy):
        '''Just to make generic code work on GenParticles'''
        return 0

    def absIso(self, dummy):
        '''Just to make generic code work on GenParticles'''
        return 0

    def absEffAreaIso(self,rho):
        '''Just to make generic code work on GenParticles'''
        return 0

    def relEffAreaIso(self,rho):
        '''Just to make generic code work on GenParticles'''
        return 0
