
class ElectronMVAID:
    def __init__(self,name,type,*xmls):
        import ROOT
        self.name = name
        self.estimator = ROOT.EGammaMvaEleEstimatorFWLite()
        self.sxmls = ROOT.vector(ROOT.string)()
        for x in xmls: self.sxmls.push_back(x)  
        self.etype = -1
        if type == "Trig":     self.etype = self.estimator.kTrig;
        if type == "NonTrig":  self.etype = self.estimator.kNonTrig;
        if type == "TrigNoIP": self.etype = self.estimator.kTrigNoIP;
        if self.etype == -1: raise RuntimeError, "Unknown type %s" % type
        self._init = False
    def __call__(self,ele,vtx,rho,full5x5=False,debug=False):
        if not self._init:
            self.estimator.initialize(self.name,self.etype,True,self.sxmls)
            self._init = True
        return self.estimator.mvaValue(ele,vtx,rho,full5x5,debug)

ElectronMVAID_Trig = ElectronMVAID("BDT", "Trig", 
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat1.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat2.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat3.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat4.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat5.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigV0_Cat6.weights.xml.gz",
)
ElectronMVAID_NonTrig = ElectronMVAID("BDT", "NonTrig", 
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat1.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat2.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat3.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat4.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat5.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_NonTrigV0_Cat6.weights.xml.gz",
)
ElectronMVAID_TrigNoIP = ElectronMVAID("BDT", "TrigNoIP", 
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat1.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat2.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat3.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat4.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat5.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat6.weights.xml.gz",
)
