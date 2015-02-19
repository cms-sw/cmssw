import copy
from PhysicsTools.Heppy.physicsobjects.Particle import Particle

class PhysicsObject(Particle):
    '''Wrapper to particle-like C++ objects.'''

    def __init__(self, physObj):
        self.physObj = physObj
        super(PhysicsObject, self).__init__()

    def __copy__(self):
        '''Very dirty trick, the physObj is deepcopied...'''
        physObj = copy.deepcopy( self.physObj )
        newone = type(self)(physObj)
        newone.__dict__.update(self.__dict__)
        newone.physObj = physObj
        return newone        

    def scaleEnergy( self, scale ):
        p4 = self.physObj.p4()
        p4 *= scale 
        self.physObj.setP4( p4 )  
        
        
    def __getattr__(self,name):
        '''Makes all attributes and methods of the wrapped physObj
        directly available.'''
        return getattr(self.physObj, name)

