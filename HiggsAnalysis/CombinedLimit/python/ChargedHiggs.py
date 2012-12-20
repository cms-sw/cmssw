from HiggsAnalysis.CombinedLimit.PhysicsModel import *

class BRChargedHiggs(PhysicsModel):
    def __init__(self):
        PhysicsModel.__init__(self)

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar('BR[0,0,1]');

        self.modelBuilder.doSet('POI','BR')

        self.modelBuilder.factory_('expr::Scaling_HH("@0*@0", BR)')
        self.modelBuilder.factory_('expr::Scaling_WH("2 * (1-@0)*@0", BR)')
        #self.modelBuilder.factory_('expr::Scaling_tt("(1-@0)*(1-@0)", BR)')
        self.modelBuilder.factory_('expr::Scaling_tt("1 - (@0+@1)", Scaling_HH, Scaling_WH)')

        self.processScaling = { 'HH':'HH', 'WH':'WH', 'HW':'WH', 'tt':'tt' }

        self.modelBuilder.out.Print()
        
    def getYieldScale(self,bin,process):

        for prefix, model in self.processScaling.iteritems():
            if process.startswith(prefix):
                return 'Scaling_'+model
            
        return 1


brChargedHiggs = BRChargedHiggs()

