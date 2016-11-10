from PhysicsTools.HeppyCore.particles.handle import Handle
from PhysicsTools.HeppyCore.particles.p4 import P4

class Jet(Handle, P4):
    pass

class Particle(Handle, P4):
        
    def __str__(self):
        tmp = '{className} : id = {id:3} pt = {pt:5.1f}, eta = {eta:5.2f}, phi = {phi:5.2f}, mass = {mass:5.2f}'
        return tmp.format(
            className = self.__class__.__name__,
            id = self.read().Core.Type,
            pt = self.read().Core.P4.Pt,
            eta = self.read().Core.P4.Eta,
            phi = self.read().Core.P4.Phi,
            mass = self.read().Core.P4.Mass
            )
    

