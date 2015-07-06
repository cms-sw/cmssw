from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.average import Average

class TauDecayModeWeighter( Analyzer ):
    '''Gets tau decay mode efficiency weight and puts it in the event'''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(TauDecayModeWeighter,self).__init__(cfg_ana, cfg_comp, looperName)

            
    def beginLoop(self, setup):
        print self, self.__class__
        super(TauDecayModeWeighter,self).beginLoop(setup)
        self.averages.add('weight', Average('weight') )
        # For 1 prong no pi-zero. Can be extended to more weights
        # if necessary
        # JAN: set to 1. until remeasured with 2015 data
        self.oneProngNoPiZeroWeight = 1.

    def process(self, event):
        self.weight = 1
        
        # RIC: let the user decide which leg needs to be corrected. In tt is both.
        # Default is leg1, so that for mt and et this modification is transparent 
        legs = self.cfg_ana.legs if hasattr(self.cfg_ana, 'legs') else ['leg1']
        
        # Not strictly correct, but this was agreed upon for Summer 2013
        # JAN: Should update to gen-matched taus
        if self.cfg_comp.isEmbed or 'Higgs' in self.cfg_comp.name or ('DY' in self.cfg_comp.name and event.isFake == 0):
            for leg in legs :
                decayMode = getattr(event.diLepton, leg)().decayMode()
                if decayMode == 0:
                    self.weight *= self.oneProngNoPiZeroWeight

            # print decayMode, self.weight, self.cfg_comp.name
            
        event.eventWeight *= self.weight
        event.tauESWeight = self.weight

        if self.cfg_ana.verbose:
            for leg in legs :            
              print 'TauDecayModeWeighter\t',leg,'\t', getattr(event.diLepton, leg)().decayMode(), event.isFake, event.tauESWeight

        self.averages['weight'].add( self.weight )
        return True
                
