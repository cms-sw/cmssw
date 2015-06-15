from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average

from CMGTools.H2TauTau.proto.TriggerEfficiency import TriggerEfficiency
from CMGTools.H2TauTau.proto.analyzers.RecEffCorrection import recEffMapEle, recEffMapMu


class LeptonWeighter( Analyzer ):
    '''Gets lepton efficiency weight and puts it in the event'''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(LeptonWeighter,self).__init__(cfg_ana, cfg_comp, looperName)

        self.leptonName = self.cfg_ana.lepton
        # self.lepton = None
        self.weight = None
        # self.weightFactor = 1.
        self.trigEff = None
        if (self.cfg_comp.isMC or self.cfg_comp.isEmbed) and \
               not ( hasattr(self.cfg_ana,'disable') and self.cfg_ana.disable is True ):
                self.trigEff = TriggerEfficiency()
                self.trigEff.lepEff = getattr( self.trigEff,
                                               self.cfg_ana.effWeight )
                self.trigEff.lepEffMC = None
                if hasattr( self.cfg_ana, 'effWeightMC'):
                    self.trigEff.lepEffMC = getattr( self.trigEff,
                                                     self.cfg_ana.effWeightMC )

            
    def beginLoop(self, setup):
        print self, self.__class__
        super(LeptonWeighter,self).beginLoop(setup)
        self.averages.add('weight', Average('weight') )
        self.averages.add('triggerWeight', Average('triggerWeight') )
        self.averages.add('eff_data', Average('eff_data') )
        self.averages.add('eff_MC', Average('eff_MC') )
        self.averages.add('recEffWeight', Average('recEffWeight') )
        self.averages.add('idWeight', Average('idWeight') )
        self.averages.add('isoWeight', Average('isoWeight') )


    def process(self, event):
        self.readCollections( event.input )
        lep = getattr( event, self.leptonName )
        lep.weight = 1
        lep.triggerWeight = 1
        lep.triggerEffData = 1
        lep.triggerEffMC = 1 
        lep.recEffWeight = 1
        lep.idWeight = 1
        lep.isoWeight = 1

        if (self.cfg_comp.isMC or self.cfg_comp.isEmbed) and \
           not ( hasattr(self.cfg_ana,'disable') and self.cfg_ana.disable is True ) and lep.pt() < 9999.:
            assert( self.trigEff is not None )
            lep.triggerEffData = self.trigEff.lepEff( lep.pt(),
                                                      lep.eta() )
            lep.triggerWeight = lep.triggerEffData

            # JAN: Don't apply MC trigger efficiency for embedded samples
            if not self.cfg_comp.isEmbed and self.trigEff.lepEffMC is not None and \
                   len(self.cfg_comp.triggers)>0:
                lep.triggerEffMC = self.trigEff.lepEffMC( lep.pt(),
                                                          lep.eta() )
                if lep.triggerEffMC>0:
                    lep.triggerWeight /= lep.triggerEffMC
                else:
                    lep.triggerWeight = 1.                    

            if hasattr( self.cfg_ana, 'idWeight'):
                lep.idWeight = self.cfg_ana.idWeight.weight(lep.pt(), abs(lep.eta()) ).weight.value
            # JAN: Do not apply iso weight for embedded sample
            if hasattr( self.cfg_ana, 'isoWeight'):
                if not self.cfg_comp.isEmbed:
                    lep.isoWeight = self.cfg_ana.isoWeight.weight(lep.pt(), abs(lep.eta()) ).weight.value
                else:
                    print 'Not applying isolation weights for embedded samples, to be reconsidered in 2015!'
            
        lep.recEffWeight = lep.idWeight * lep.isoWeight
        lep.weight = lep.triggerWeight * lep.recEffWeight

        event.eventWeight *= lep.weight
	if not hasattr(event,"triggerWeight"): event.triggerWeight=1.0
        event.triggerWeight *= lep.triggerWeight
        self.averages['weight'].add( lep.weight )
        self.averages['triggerWeight'].add( lep.triggerWeight )
        self.averages['eff_data'].add( lep.triggerEffData )
        self.averages['eff_MC'].add( lep.triggerEffMC )
        self.averages['recEffWeight'].add( lep.recEffWeight )
        self.averages['idWeight'].add( lep.idWeight )
        self.averages['isoWeight'].add( lep.isoWeight )
                
