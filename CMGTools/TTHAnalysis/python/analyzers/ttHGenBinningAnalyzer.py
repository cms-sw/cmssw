import operator 
import itertools
import copy

#from ROOT import TLorentzVector

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

#from CMGTools.RootTools.physicsobjects.genutils import *
        
class ttHGenBinningAnalyzer( Analyzer ):
    """
    Add the Gen Level binning quantities to the event for validation of MC reweighting
    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHGenBinningAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------
        

    def declareHandles(self):
        super(ttHGenBinningAnalyzer, self).declareHandles()

        self.mchandles['genInfo'] = AutoHandle( 'generator', 'GenEventInfoProduct' )

    def beginLoop(self, setup):
        super(ttHGenBinningAnalyzer,self).beginLoop( setup )

    def addGenBinning(self,event):
        if self.mchandles['genInfo'].product().hasBinningValues():
            event.genBin = self.mchandles['genInfo'].product().binningValues()[0]
        else:
            event.genBin = -999

        event.genQScale = self.mchandles['genInfo'].product().qScale()

    def process(self, event):
        self.readCollections( event.input )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        self.addGenBinning(event)
        return True

