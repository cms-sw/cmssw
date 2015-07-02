import re

from ROOT import TFile

from PhysicsTools.HeppyCore.statistics.average  import Average
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class HiggsPtWeighter( Analyzer ):
    '''Weight the event to get the NLO Higgs pT distribution for ggH events'''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(HiggsPtWeighter,self).__init__(cfg_ana, cfg_comp, looperName)
        
        self.rePatGGH = re.compile('HiggsGGH(\d+)')
        
        self.isGGH, self.higgsMass = self.isGGH(self.cfg_comp)
        if self.isGGH:
            self.higgsPtWeightFile = TFile("$CMSSW_BASE/src/CMGTools/H2TauTau/data/weight_ptH_{mass}_8TeV.root".format(mass=self.higgsMass))
            self.higgsPtWeightHistogram     = self.higgsPtWeightFile.Get("Nominal")
            self.higgsPtWeightHistogramUp   = self.higgsPtWeightFile.Get("Up"     )
            self.higgsPtWeightHistogramDown = self.higgsPtWeightFile.Get("Down"   )
   
    def isGGH(self, component):
        if self.cfg_comp.isMC:
            match = self.rePatGGH.match( self.cfg_comp.name )
            if not match:
                return False, -1
            else:
                mass = match.group(1)
                return True, mass
        else:
            return False, -1
        
    def beginLoop(self, setup):
        print self, self.__class__
        super(HiggsPtWeighter,self).beginLoop(setup)
        self.averages.add('weight', Average('weight') )

    def process(self, event):

        event.higgsPtWeight     = 1 
        event.higgsPtWeightUp   = 1 
        event.higgsPtWeightDown = 1 

        if not self.isGGH: return True
        
        higgsBosons = event.genHiggsBosons
        
        if len(higgsBosons)!=1:
            strerr = '{nhiggs} Higgs bosons, this should not happen for a ggH component. \
                      Your component is:\n {comp}'.format(nhiggs=len(higgsBosons), comp=str(self.cfg_comp))
            raise ValueError(strerr)

        event.genHiggs = higgsBosons[0] 
        higgsPt = event.genHiggs.pt()
        event.higgsPtWeight     = self.higgsPtWeightHistogram    .GetBinContent(self.higgsPtWeightHistogram    .FindBin(higgsPt))
        event.higgsPtWeightUp   = self.higgsPtWeightHistogramUp  .GetBinContent(self.higgsPtWeightHistogramUp  .FindBin(higgsPt))
        event.higgsPtWeightDown = self.higgsPtWeightHistogramDown.GetBinContent(self.higgsPtWeightHistogramDown.FindBin(higgsPt))

        ### MULTIPLYING TO THE EVENT WEIGHT 
        event.eventWeight *= event.higgsPtWeight
        ### 
        
        self.averages['weight'].add( event.higgsPtWeight )
        if self.cfg_ana.verbose:
            print 'HiggsPtWeighter: pt_H={pt:6.2f}, weight={weight:6.2f}'.format(
                pt    = higgsPt,
                weight= event.higgsPtWeight
            )
        return True

