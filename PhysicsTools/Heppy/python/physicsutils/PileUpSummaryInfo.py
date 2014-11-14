class PileUpSummaryInfo( object ):
    def __init__(self, object ):
        self.object = object
        
    def __getattr__(self,name):
        '''all accessors  from cmg::DiTau are transferred to this class.'''
        return getattr(self.object, name)

    def nPU(self):
        return self.object.getPU_NumInteractions()
    
    def nTrueInteractions(self):
        return self.object.getTrueNumInteractions()
        
    def __str__(self):
        tmp = '{className} : bunchx = {bunchx}; numPU = {numpu}'.format(
            className = self.__class__.__name__,
            bunchx = self.object.getBunchCrossing(),
            numpu = self.object.getPU_NumInteractions() )
        return tmp

