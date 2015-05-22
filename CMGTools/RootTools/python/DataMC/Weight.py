from CMGTools.RootTools.PersistentDict import PersistentDict

class Weight( PersistentDict ):

    def __init__(self, name, fileName):
        '''Read weight information from a weight file. see self.GetWeight'''
        PersistentDict.__init__(self, name, fileName)

    def GetWeight(self):
        '''Return the weight, and fill weight-related attributes'''
        self.genNEvents = float(self.Value('genNEvents'))
        self.genEff = float(self.Value('genEff'))
        self.xSection = float(self.Value('xSection'))
        self.intLumi  = float(self.Value('intLumi'))
        return self.xSection * self.intLumi / ( self.genNEvents * self.genEff) 

    def SetIntLumi(self, lumi):
        '''Set integrated luminosity.'''
        self.dict['intLumi'] = lumi

    def __str__(self):
        return PersistentDict.__str__(self) + ' weight = %3.5f' % self.GetWeight() 
