from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

import PhysicsTools.HeppyCore.framework.config as cfg

        
class HiggsDecayModeAnalyzer( Analyzer ):
    """Classify and filter events according to Higgs boson decays

       Reads:
         event.genHiggsBosons

       Creates in the event:
         event.genHiggsDecayMode =   0  for non-Higgs or multi-higgs
                                 15  for H -> tau tau
                                 23  for H -> Z Z
                                 24  for H -> W W
                                 xx  for H -> xx yy zzz 
       If filterHiggsDecays is set to a list of Higgs decay modes,
       it will filter events that have those decay modes.
       e.g. [0, 15, 23, 24] will keep data, non-Higgs MC and Higgs decays to (tau, Z, W) 
       but will drop Higgs decays to other particles (e.g. bb).
      
    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(HiggsDecayModeAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
    #---------------------------------------------
        

    def declareHandles(self):
        super(HiggsDecayModeAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(HiggsDecayModeAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        higgsBosons = event.genHiggsBosons
        if len(higgsBosons) != 1:
            event.genHiggsDecayMode = 0
            event.genHiggsBoson     = None
        else:
            event.genHiggsBoson = higgsBosons[0]
            event.genHiggsDecayMode = abs( event.genHiggsBoson.daughter(0).pdgId() )

        # if MC and filtering on the Higgs decay mode, 
        # them do filter events
        if self.cfg_ana.filterHiggsDecays:
            if event.genHiggsDecayMode not in self.cfg_ana.filterHiggsDecays:
                return False

        return True

setattr(HiggsDecayModeAnalyzer,"defaultConfig",
    cfg.Analyzer(HiggsDecayModeAnalyzer,
        filterHiggsDecays = False, 
    )
)
