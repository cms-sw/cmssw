import ROOT 

class ElectronMVAID:
    def __init__(self,name,type,*xmls):
        self.name = name
        self.estimator = ROOT.heppy.EGammaMvaEleEstimatorFWLite() 
        self.sxmls = ROOT.vector(ROOT.string)()
        for x in xmls: self.sxmls.push_back(x)  
        self.etype = -1
        if type == "Trig":     self.etype = self.estimator.kTrig;
        if type == "NonTrig":  self.etype = self.estimator.kNonTrig;
        if type == "TrigNoIP": self.etype = self.estimator.kTrigNoIP;
        if type == "TrigCSA14":     self.etype = self.estimator.kTrigCSA14;
        if type == "NonTrigCSA14":  self.etype = self.estimator.kNonTrigCSA14;
        if type == "NonTrigPhys14":  self.etype = self.estimator.kNonTrigPhys14;
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

ElectronMVAID_TrigCSA14bx50 = ElectronMVAID("BDT", "TrigCSA14", 
        "EgammaAnalysis/ElectronTools/data/CSA14/TrigIDMVA_50ns_EB_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/TrigIDMVA_50ns_EE_BDT.weights.xml.gz",
)
ElectronMVAID_TrigCSA14bx25 = ElectronMVAID("BDT", "TrigCSA14", 
        "EgammaAnalysis/ElectronTools/data/CSA14/TrigIDMVA_25ns_EB_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/TrigIDMVA_25ns_EE_BDT.weights.xml.gz",
)

ElectronMVAID_NonTrigCSA14bx25 = ElectronMVAID("BDT", "NonTrigCSA14", 
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EB_5_25ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EE_5_25ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EB_10_25ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EE_10_25ns_BDT.weights.xml.gz",
)
ElectronMVAID_NonTrigCSA14bx50 = ElectronMVAID("BDT", "NonTrigCSA14", 
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EB_5_50ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EE_5_50ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EB_10_50ns_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/CSA14/EIDmva_EE_10_50ns_BDT.weights.xml.gz",
)

ElectronMVAID_NonTrigPhys14 = ElectronMVAID("BDT", "NonTrigPhys14", 
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EB1_5_oldscenario2phys14_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EB2_5_oldscenario2phys14_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EE_5_oldscenario2phys14_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EB1_10_oldscenario2phys14_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EB2_10_oldscenario2phys14_BDT.weights.xml.gz",
        "EgammaAnalysis/ElectronTools/data/PHYS14/EIDmva_EE_10_oldscenario2phys14_BDT.weights.xml.gz",
)

ElectronMVAID_ByName = {
    'Trig':ElectronMVAID_Trig,
    'NonTrig':ElectronMVAID_NonTrig,
    'TrigNoIP':ElectronMVAID_TrigNoIP,
    'TrigCSA14bx50':ElectronMVAID_TrigCSA14bx50,
    'TrigCSA14bx25':ElectronMVAID_TrigCSA14bx25,
    'NonTrigCSA14bx25':ElectronMVAID_NonTrigCSA14bx25,
    'NonTrigCSA14bx50':ElectronMVAID_NonTrigCSA14bx50,
    'NonTrigPhys14':ElectronMVAID_NonTrigPhys14,
}
