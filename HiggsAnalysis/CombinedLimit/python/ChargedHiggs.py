from HiggsAnalysis.CombinedLimit.PhysicsModel import *

class BRChargedHiggs(PhysicsModel):
    def __init__(self):
        PhysicsModel.__init__(self)

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar('BR[0.05,0,1]');

        self.modelBuilder.doSet('POI','BR')

        self.modelBuilder.factory_('expr::Scaling_HH("@0*@0", BR)')
        self.modelBuilder.factory_('expr::Scaling_WH("2 * (1-@0)*@0", BR)')
        #self.modelBuilder.factory_('expr::Scaling_tt("(1-@0)*(1-@0)", BR)')
        self.modelBuilder.factory_('expr::Scaling_tt("1 - (@0+@1)", Scaling_HH, Scaling_WH)')

        self.processScaling = [ 'HH', 'WH', 'tt' ]

        self.modelBuilder.out.Print()
        
    def getYieldScale(self,bin,process):

        print bin, process

        for s in self.processScaling:
            if s in process:
                return 'Scaling_'+s
            
        return 1


brChargedHiggs = BRChargedHiggs()

