# This extracts the constants formerly used in lumiQueryAPI.py so that they can be used in the pileup
# scripts.

class ParametersObject(object):

    def __init__ (self):
        self.NBX               = 3564
        self.bunchSpacing      = 24.95e-9
        self.orbitLength       = self.NBX*self.bunchSpacing
        self.orbitFrequency    = 1.0/self.orbitLength
        self.orbitsPerLS       = 2**18
        self.lumiSectionLength = self.orbitsPerLS*self.orbitLength
