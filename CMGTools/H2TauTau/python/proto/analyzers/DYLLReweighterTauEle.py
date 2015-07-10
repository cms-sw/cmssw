from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class DYLLReweighterTauEle( Analyzer ):
    '''Apply the reweighting calculated by Jose on the basis of the data/mc agreement
       in the inclusive sample, see here:
       https://indico.cern.ch/getFile.py/access?contribId=38&resId=0&materialId=slides&confId=212612
       event.zllWeight is added to the event and multiplied to event.eventWeight
    '''

    def process(self, event):
        event.zllWeight = 1
        if not self.cfg_comp.isMC:
            return True

        # do nothing in all cases, but the DY -> ll
        if event.isFake != 1 or self.cfg_comp.name.find('DY') == -1 :
            return True
            
        tau = event.diLepton.leg1()
        if tau.decayMode() == 0 :   # 1prong 0pi
            if abs (tau.eta()) < 1.5 :
                event.zllWeight = self.cfg_ana.W1p0PB 
            else:
                event.zllWeight = self.cfg_ana.W1p0PE 
        elif tau.decayMode() == 1 : # 1prong 1pi
            if abs (tau.eta()) < 1.5 :
                event.zllWeight = self.cfg_ana.W1p1PB 
            else:
                event.zllWeight = self.cfg_ana.W1p1PE 

        if self.cfg_ana.verbose:
            print 'DYLLReweighterTauEle',tau.decayMode(),tau.eta(),event.zllWeight
        
        event.eventWeight = event.eventWeight * event.zllWeight
        return True

# FIXME read from cfg file the scaling factors
