from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy

from CMGTools.H2TauTau.proto.analyzers.varsDictionary import vars as var_dict
from CMGTools.H2TauTau.proto.analyzers.TreeVariables import event_vars, ditau_vars, particle_vars, lepton_vars, electron_vars, muon_vars, tau_vars, jet_vars, geninfo_vars, vbf_vars

class H2TauTauTreeProducerBase(TreeAnalyzerNumpy):

    '''
       Base H->tautau tree producer.
       Provides basic functionality for tau-tau specific trees.

       The branch names can be changed by means of a dictionary.
    '''

    def __init__(self, *args):
        super(H2TauTauTreeProducerBase, self).__init__(*args)
        self.varStyle = 'std'
        self.varDict = var_dict
        self.skimFunction = 'True'
        if hasattr(self.cfg_ana, 'varStyle'):
            self.varStyle = self.cfg_ana.varStyle
        if hasattr(self.cfg_ana, 'varDict'):
            self.varDict = self.cfg_ana.varDict
        if hasattr(self.cfg_ana, 'skimFunction'):
            self.skimFunction = self.cfg_ana.skimFunction

    def var(self, tree, varName, type=float):
        tree.var(self.varName(varName), type)

    def fill(self, tree, varName, value):
        tree.fill(self.varName(varName), value)

    def varName(self, name):
        try:
            return self.varDict[name][self.varStyle]
        except:
            if self.verbose:
                print 'WARNING: self.varDict[{NAME}][{VARSTYLE}] does not exist'.format(NAME=name, VARSTYLE=self.varStyle)
                print '         using {NAME}'.format(NAME=name)
            return name

    def fillTree(self, event):
        if eval(self.skimFunction):
            self.tree.tree.Fill()

    def bookGeneric(self, tree, var_list, obj_name=None):
        for var in var_list:
            names = [obj_name, var.name] if obj_name else [var.name]
            self.var(tree, '_'.join(names), var.type)

    def fillGeneric(self, tree, var_list, obj, obj_name=None):
        for var in var_list:
            names = [obj_name, var.name] if obj_name else [var.name]
            self.fill(tree, '_'.join(names), var.function(obj))


    def declareVariables(self, setup):
        ''' Declare all variables here in derived calss
        '''
        pass

    def process(self, event):
        ''' Fill variables here in derived class

        End implementation with self.fillTree(event)
        '''
        # needed when doing handle.product(), goes back to
        # PhysicsTools.Heppy.analyzers.core.Analyzer
        self.tree.reset()

        if not eval(self.skimFunction):
            return False

        # self.fillTree(event)

    # event
    def bookEvent(self, tree):
        self.bookGeneric(tree, event_vars)

    def fillEvent(self, tree, event):
        self.fillGeneric(tree, event_vars, event)

    # simple particle
    def bookParticle(self, tree, p_name):
        self.bookGeneric(tree, particle_vars, p_name)

    def fillParticle(self, tree, p_name, particle):
        self.fillGeneric(tree, particle_vars, particle, p_name)

    # simple gen particle
    def bookGenParticle(self, tree, p_name):
        self.bookParticle(tree, p_name)
        self.var(tree, '{p_name}_pdgId'.format(p_name=p_name))

    def fillGenParticle(self, tree, p_name, particle):
        self.fillParticle(tree, p_name, particle)
        self.fill(tree, '{p_name}_pdgId'.format(p_name=p_name), particle.pdgId())

    # di-tau
    def bookDiLepton(self, tree):
        # RIC: to add
        # svfit 'fittedDiTauSystem', 'fittedMET', 'fittedTauLeptons'
        self.bookGeneric(tree, ditau_vars)
        self.bookParticle(tree, 'svfit_l1')
        self.bookParticle(tree, 'svfit_l2')

    def fillDiLepton(self, tree, diLepton):
        self.fillGeneric(tree, ditau_vars, diLepton)
        if hasattr(diLepton, 'svfit_Taus'):
            for i, tau in enumerate(diLepton.svfitTaus()):
                self.fillParticle(tree, 'svfit_l' + str(i + 1), tau)

    # lepton
    def bookLepton(self, tree, p_name):
        self.bookParticle(tree, p_name)
        self.bookParticle(tree, p_name + '_jet')
        self.bookGeneric(tree, lepton_vars, p_name)

    def fillLepton(self, tree, p_name, lepton):
        self.fillParticle(tree, p_name, lepton)
        if hasattr(lepton, 'jet'):
            self.fillParticle(tree, p_name + '_jet', lepton.jet)
        self.fillGeneric(tree, lepton_vars, lepton, p_name)

    # muon
    def bookMuon(self, tree, p_name):
        self.bookLepton(tree, p_name)
        self.bookGeneric(tree, muon_vars, p_name)

    def fillMuon(self, tree, p_name, muon):
        self.fillLepton(tree, p_name, muon)
        self.fillGeneric(tree, muon_vars, muon, p_name)

    # ele
    def bookEle(self, tree, p_name):
        self.bookLepton(tree, p_name)
        self.bookGeneric(tree, electron_vars, p_name)

    def fillEle(self, tree, p_name, ele):
        self.fillLepton(tree, p_name, ele)
        self.fillGeneric(tree, electron_vars, ele, p_name)

    # tau
    def bookTau(self, tree, p_name):
        self.bookLepton(tree, p_name)
        self.bookGeneric(tree, tau_vars, p_name)

    def fillTau(self, tree, p_name, tau):
        self.fillLepton(tree, p_name, tau)
        self.fillGeneric(tree, tau_vars, tau, p_name)

    # jet
    def bookJet(self, tree, p_name):
        self.bookParticle(tree, p_name)
        self.bookGeneric(tree, jet_vars, p_name)
        
    def fillJet(self, tree, p_name, jet):
        self.fillParticle(tree, p_name, jet)
        self.fillGeneric(tree, jet_vars, jet, p_name)

    # vbf
    def bookVBF(self, tree, p_name):
        self.bookGeneric(tree, vbf_vars, p_name)

    def fillVBF(self, tree, p_name, vbf):
        self.fillGeneric(tree, vbf_vars, vbf, p_name)

    # generator information
    def bookGenInfo(self, tree):
        self.bookGeneric(tree, geninfo_vars)

    def fillGenInfo(self, tree, event):
        self.fillGeneric(tree, geninfo_vars, event)

    # quark and gluons
    def bookQG(self, tree, maxNGenJets=2):
        for i in range(0, maxNGenJets):
            self.bookGenParticle(self.tree, 'genqg_{i}'.format(i=i))

    def fillQG(self, tree, event, maxNGenJets=2):
        # Fill hard quarks/gluons
        quarksGluons = [p for p in event.genParticles if abs(p.pdgId()) in (1, 2, 3, 4, 5, 21) and
                        p.status() == 3 and
                        (p.numberOfDaughters() == 0 or p.daughter(0).status() != 3)]
        quarksGluons.sort(key=lambda x: -x.pt())
        for i in range(0, min(maxNGenJets, len(quarksGluons))):
            self.fillGenParticle(
                tree, 'genqg_{i}'.format(i=i), quarksGluons[i])
