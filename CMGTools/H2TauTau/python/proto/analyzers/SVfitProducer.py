from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from TauAnalysis.SVfitStandalone.SVfitStandaloneAlgorithm import SVfitAlgo
from TauAnalysis.SVfitStandalone.MeasuredTauLepton import measuredTauLepton

from ROOT import TMatrixD, std

class SVfitProducer(Analyzer):
    '''Computes SVfit di-tau mass at the ntuple level.'''

    def __init__(self, *args):
        super(SVfitProducer, self).__init__(*args)
        self.legType = {'undef':0, 'tau':1, 'ele':2, 'muon':3, 'prompt':4}

    def process(self, event):

        decayMode1 = -1
        decayMode2 = -1

        if self.cfg_ana.l1type == 'tau':
            decayMode1 = event.leg1.decayMode()
        if self.cfg_ana.l2type == 'tau':
            decayMode2 = event.leg2.decayMode()

        # RIC: some PF muons/electron can get the wrong mass assigned.
        # Peg their masses to the PDG values
        if   self.cfg_ana.l1type == 'muon' : mass1 = 0.10566    # PDG mass [GeV]
        elif self.cfg_ana.l1type == 'ele'  : mass1 = 0.51100e-3 # PDG mass [GeV]
        else : mass1 = event.leg1.mass()

        if   self.cfg_ana.l2type == 'muon' : mass2 = 0.10566    # PDG mass [GeV]
        elif self.cfg_ana.l2type == 'ele'  : mass2 = 0.51100e-3 # PDG mass [GeV]
        else : mass2 = event.leg2.mass()

        leg1 = measuredTauLepton(self.legType[self.cfg_ana.l1type], event.leg1.pt(),
                                 event.leg1.eta(), event.leg1.phi(), mass1, decayMode1)
        leg2 = measuredTauLepton(self.legType[self.cfg_ana.l2type], event.leg2.pt(),
                                 event.leg2.eta(), event.leg2.phi(), mass2, decayMode2)

        measuredLeptons = std.vector('svFitStandalone::MeasuredTauLepton')()

        # RIC: not really needed since SVfit internally sorts the inputs
        if hasattr(self.cfg_ana, 'order') and self.cfg_ana.order == '12' :
            measuredLeptons.push_back(leg1)
            measuredLeptons.push_back(leg2)
        else :
            measuredLeptons.push_back(leg2)
            measuredLeptons.push_back(leg1)

        metcov = TMatrixD(2, 2)

        metcov[0][0] = event.diLepton.mvaMetSig(0, 0)
        metcov[0][1] = event.diLepton.mvaMetSig(0, 1)
        metcov[1][0] = event.diLepton.mvaMetSig(1, 0)
        metcov[1][1] = event.diLepton.mvaMetSig(1, 1)

        mex = event.diLepton.met().px()
        mey = event.diLepton.met().py()

        svfit = SVfitAlgo(measuredLeptons, mex, mey, metcov, 2*self.cfg_ana.verbose)

        # add an additional logM(tau,tau) term to the nll
        # to suppress tails on M(tau,tau) (default is false)
        # svfit.addLogM(False)

        if self.cfg_ana.integration == 'VEGAS':
            svfit.integrateVEGAS()
        elif self.cfg_ana.integration == 'MarkovChain':
            svfit.integrateMarkovChain()
        else:
            print 'The integration method must be defined in the cfg as "integration".'
            print 'Options: [VEGAS, MarkovChain]'
            raise

        # debug
        if self.cfg_ana.verbose:
            if abs(event.diLepton.svfitMass()-svfit.mass()) > 0.01:
                print 'WARNING: run {RUN}, lumi {LUMI}, event {EVT}'.format(RUN=str(event.run),
                                                                            LUMI=str(event.lumi),
                                                                            EVT=str(event.eventId))
                print 'precomputed svfit mass   ', event.diLepton.svfitMass()
                print 'svfit mass computed here ', svfit.mass()

        # method override
        event.diLepton.svfitMass      = svfit.mass
        event.diLepton.svfitMassError = svfit.massUncert

        # add also the pt, eta and phi as computed by SVfit
        if self.cfg_ana.integration == 'MarkovChain':
            event.diLepton.svfitPt       = svfit.pt
            event.diLepton.svfitPtError  = svfit.ptUncert
            event.diLepton.svfitEta      = svfit.eta
            event.diLepton.svfitPhi      = svfit.phi
            event.diLepton.svfitMET      = svfit.fittedMET
            event.diLepton.svfitTaus     = svfit.fittedTauLeptons
