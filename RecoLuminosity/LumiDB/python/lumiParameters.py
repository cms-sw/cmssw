class ParametersObject (object):
    '''
    collection of constants used in lumi related calculation
    '''
    def __init__ (self):
        self.NBX             = 3564  # number beam crossings
        self.numorbit        = 2**18 # 262144
        self.rotationRate    = 11245.613 # for 3.5 TeV Beam energy
        self.rotationTime    = 1 / self.rotationRate
        self.lumiSectionLen  = self.numorbit * self.rotationTime
        ##self.minBiasXsec   = 71300 # unit: microbarn

        
    def setRotationRate(self,rate):
        '''
        update the default LHC orbit frequency
        Single beam energy of 450GeV:  11245.589
        Single beam energy of 3.5TeV: 11245.613
        '''
        self.rotationRate =rate
        
    def setNumOrbit(self,numorbit):
        self.numorbit=numorbit
        
    def setNumBx(self,numbx):
        '''
        update the default number of BX
        '''
        self.NBX = numbx
        
    def calculateTimeParameters(self):
        '''Given the rotation rate, calculate lumi section length and
        rotation time.  This should be called if rotationRate is
        updated.
        '''
        self.rotationTime    = 1 / self.rotationRate
        self.lumiSectionLen  = self.numorbit * self.rotationTime
        
    def lslengthsec(self):
        '''
        Calculate lslength in sec from number of orbit and BX
        '''
        return self.lumiSectionLen 
       
#=======================================================
#   Unit Test
#=======================================================
if __name__ == "__main__":
    p=ParametersObject()
    print p.lslengthsec()
    print p.NBX
    print p.numorbit
    
