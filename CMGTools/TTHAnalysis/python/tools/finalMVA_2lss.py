from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *

class FinalMVA_2LSS:
    def __init__(self):
        self._MVAs = {}
        self._vars_1_6 = [ 
                #MVAVar("lep2AbsEta", func = lambda ev : min(abs(ev.LepGood1_eta),abs(ev.LepGood2_eta))),
                MVAVar("lep2AbsEta", func = lambda ev : abs(ev.LepGood2_eta)),
                MVAVar("lep2Pt",     func = lambda ev : ev.LepGood2_pt),
                MVAVar("MHT",        func = lambda ev : ev.mhtJet25),
                MVAVar("mindr_lep2_jet", func = lambda ev : ev.mindr_lep2_jet),
                MVAVar("MT_met_lep1",    func = lambda ev : ev.MT_met_lep1),
                MVAVar("sum_pt",         func = lambda ev : ev.htJet25) 
        ]
        self._vars_7_9 = [
                MVAVar("avg_dr_jets",      func = lambda ev : ev.avg_dr_jet),
                MVAVar("mindr_lep1_jet",   func = lambda ev : ev.mindr_lep1_jet),
                MVAVar("MT_met_leplep",    func = lambda ev : ev.MT_met_leplep),
        ]
        self._var_10 = [
                MVAVar("numJets_float",    func = lambda ev : ev.nJet25)
        ]
        self._vars_11_15 = [
                MVAVar("b1_jet_pt",      func = lambda ev : ev.Jet1_pt),
                MVAVar("b2_jet_pt",      func = lambda ev : ev.Jet2_pt),
                MVAVar("lep1Pt",         func = lambda ev : ev.LepGood1_pt),
                MVAVar("sum_pt-(sum_pz-abs(pz_of_everything))",    func = lambda ev : ev.htJet25 - (ev.sum_abspz - abs(ev.sum_sgnpz))),
                MVAVar("sum_pt/sum_pz",    func = lambda ev : ev.htJet25/ev.sum_abspz),
        ]
        P="/afs/cern.ch/user/a/abrinke1/public/MultiLepton/BDT_weights/";
        self._MVAs["MVA_2LSS_23j_6var"] = MVATool("MVA_2LSS_23j_6var", 
            P+"SS_eq3jge1t_useSide_2_6var_test/TMVAClassification_BDTG.weights.xml",
            self._vars_1_6)
        self._MVAs["MVA_2LSS_23j_9var"] = MVATool("MVA_2LSS_23j_9var", 
            P+"SS_eq3jge1t_useSide_2_9var_test/TMVAClassification_BDTG.weights.xml",
            self._vars_1_6 + self._vars_7_9)
        self._MVAs["MVA_2LSS_4j_6var"] = MVATool("MVA_2LSS_4j_6var", 
            P+"SS_ge4jge1t_useSide_2_6var_test/TMVAClassification_BDTG.weights.xml",
            self._vars_1_6)
        self._MVAs["MVA_2LSS_4j_10var"] = MVATool("MVA_2LSS_4j_10var", 
            P+"SS_ge4jge1t_useSide_2_10var_test/TMVAClassification_BDTG.weights.xml",
            self._vars_1_6 + self._vars_7_9 + self._var_10)
        self._MVAs["MVA_2LSS_4j_15var"] = MVATool("MVA_2LSS_4j_15var", 
            P+"SS_ge4jge1t_useSide_2_15var_test/TMVAClassification_BDTG.weights.xml",
            self._vars_1_6 + self._vars_7_9 + self._var_10 + self._vars_11_15)
        self._MVAs["MVA_2LSS_4j_6var_cat"] = CategorizedMVA(
            [ ( lambda ev: abs(ev.LepGood1_pdgId) == 11 and abs(ev.LepGood2_pdgId) == 11,
                    MVATool("ee", P+"SS_ge4jge1t_useSide_2_6var_TwoEle/TMVAClassification_BDTG.weights.xml", self._vars_1_6) ),
              ( lambda ev: abs(ev.LepGood1_pdgId) == 13 and abs(ev.LepGood2_pdgId) == 13,
                    MVATool("ee", P+"SS_ge4jge1t_useSide_2_6var_TwoMuon/TMVAClassification_BDTG.weights.xml", self._vars_1_6) ),
              ( lambda ev: abs(ev.LepGood1_pdgId) != abs(ev.LepGood2_pdgId),
                    MVATool("ee", P+"SS_ge4jge1t_useSide_2_6var_MuonEle/TMVAClassification_BDTG.weights.xml", self._vars_1_6) ) ]
        )
    def listBranches(self):
        return self._MVAs.keys()
    def __call__(self,event):
        return dict([ (name, mva(event)) for name, mva in self._MVAs.iteritems()])

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    tree.AddFriend("sf/t", argv[2])
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = FinalMVA_2LSS()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

