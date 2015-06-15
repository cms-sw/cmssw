import operator

from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
from PhysicsTools.Heppy.physicsobjects.Electron import Electron

from CMGTools.H2TauTau.proto.analyzers.DiLeptonAnalyzer import DiLeptonAnalyzer
from CMGTools.H2TauTau.proto.physicsobjects.DiObject import TauElectron


class TauEleAnalyzer(DiLeptonAnalyzer):

    DiObjectClass = TauElectron
    LeptonClass = Electron
    OtherLeptonClass = Muon

    def declareHandles(self):
        super(TauEleAnalyzer, self).declareHandles()
        self.handles['diLeptons'] = AutoHandle(
            'cmgTauEleCorSVFitFullSel',
            'std::vector<pat::CompositeCandidate>'
        )

        self.handles['otherLeptons'] = AutoHandle(
            'slimmedMuons',
            'std::vector<pat::Muon>'
        )

        self.handles['leptons'] = AutoHandle(
            'slimmedElectrons',
            'std::vector<pat::Electron>'
        )

        self.mchandles['genParticles'] = AutoHandle('prunedGenParticles',
                                                    'std::vector<reco::GenParticle>')

        self.relaxEleId = False
        self.relaxTauId = False

    def buildDiLeptons(self, cmgDiLeptons, event):
        '''Build di-leptons, associate best vertex to both legs,
        select di-leptons with a tight ID electron.
        The electron ID selection is done so that dxy and dz can be computed
        '''
        diLeptons = []
        for index, dil in enumerate(cmgDiLeptons):
            pydil = self.__class__.DiObjectClass(dil)
            pydil.leg1().associatedVertex = event.goodVertices[0]
            pydil.leg2().associatedVertex = event.goodVertices[0]
            pydil.leg2().rho = event.rho
            pydil.mvaMetSig = pydil.met().getSignificanceMatrix()
            diLeptons.append(pydil)
        return diLeptons

    def buildLeptons(self, cmgLeptons, event):
        '''Build electrons for veto, associate best vertex, select loose ID electrons.
        Since the electrons are used for veto, the 0.3 default isolation cut is left there,
        as well as the pt 15 gev cut'''
        leptons = []
        for index, lep in enumerate(cmgLeptons):
            pyl = self.__class__.LeptonClass(lep)
            pyl.associatedVertex = event.goodVertices[0]
            pyl.rho = event.rho
            # if not pyl.looseIdForEleTau():
            #     continue
            if pyl.relIso(dBetaFactor=0.5, allCharged=0) > 0.3:
                continue
            leptons.append(pyl)
        return leptons

    def testMuonIDLoose(self, muon):
        '''Loose muon ID and kine, no isolation requirement, for lepton veto'''
        return muon.pt() > 15 and \
            abs(muon.eta()) < 2.4 and \
            muon.isGlobalMuon() and \
            muon.isTrackerMuon() and \
            muon.sourcePtr().userFloat('isPFMuon') and \
            abs(muon.dz()) < 0.2
        # self.testVertex( muon )

    def buildOtherLeptons(self, cmgOtherLeptons, event):
        '''Build muons for third lepton veto, associate best vertex.
        '''
        otherLeptons = []
        for index, lep in enumerate(cmgOtherLeptons):
            pyl = self.__class__.OtherLeptonClass(lep)
            pyl.associatedVertex = event.goodVertices[0]
            # if not self.testMuonIDLoose(pyl):
            #     continue
            otherLeptons.append(pyl)
        return otherLeptons

    def process(self, event):

        result = super(TauEleAnalyzer, self).process(event)
        import pdb; pdb.set_trace()

        event.isSignal = False

        if result is False:
            # trying to get a dilepton from the control region.
            # it must have well id'ed and trig matched legs,
            # and di-lepton veto must pass
            # i.e. only the iso requirement is relaxed

            # Also relax IDs for sideband studies without changing the
            # dilepton selectionSequence
            self.relaxEleId = True
            self.relaxTauId = True
            result = self.selectionSequence(event, fillCounter=False,
                                            leg1IsoCut=self.cfg_ana.looseiso1,
                                            leg2IsoCut=self.cfg_ana.looseiso2)
            self.relaxEleId = False
            self.relaxTauId = False

            if result is False:
                # really no way to find a suitable di-lepton,
                # even in the control region
                return False
            event.isSignal = False
        else:
            event.isSignal = event.leptonAccept and event.thirdLeptonVeto

        return True

    def testTauVertex(self, lepton):
        '''Tests vertex constraints, for tau'''
        # Just checks if the primary vertex the tau was reconstructed with
        # corresponds to the one used in the analysis
        isPV = lepton.vertex().z() == lepton.associatedVertex.z()
        return isPV

    def testVertex(self, lepton):
        '''Tests vertex constraints, for mu'''
        return abs(lepton.dxy()) < 0.045 and abs(lepton.dz()) < 0.2

    def testTauVertex(self, lepton):
        '''Tests vertex constraints, for tau'''
        isPV = lepton.vertex().z() == lepton.associatedVertex.z()
        return isPV

    def testLeg1ID(self, tau):
        # Don't apply anti-e discriminator for relaxed tau ID
        # RIC: 9 March 2015
        if self.relaxTauId:
            return ( (tau.tauID('decayModeFinding')         > 0.5  or
                      tau.tauID('decayModeFindingNewDMs')   > 0.5) and
                     tau.tauID('againstMuonLoose3')         > 0.5  and
                     (tau.zImpact() > 0.5 or tau.zImpact() < -1.5) and
                     self.testTauVertex(tau) )
        return ( (tau.tauID('decayModeFinding')         > 0.5  or
                  tau.tauID('decayModeFindingNewDMs')   > 0.5) and
                 tau.tauID('againstElectronTightMVA5')  > 0.5  and
                 tau.tauID('againstMuonLoose3')         > 0.5  and
                 (tau.zImpact() > 0.5 or tau.zImpact() < -1.5) and
                 self.testTauVertex(tau) )

    def testLeg1Iso(self, tau, isocut):
        '''if isocut is None, returns true if three-hit iso cut is passed.
        Otherwise, returns true if iso MVA > isocut.'''
        if isocut is None:
            return tau.tauID('byLooseCombinedIsolationDeltaBetaCorr3Hits') > 0.5
        else:
            # JAN FIXME - placeholder, as of now only used to define passing cuts
            # return tau.tauID("byIsolationMVA3newDMwLTraw") > isocut
            # RIC: 9 March 2015
            return tau.tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits") < isocut

    def testLeg2ID(self, electron):
        '''Tight electron selection, no isolation requirement.
           Electron ID: NonTrigPhys14, Tight
        '''
        return electron.electronID('POG_MVA_ID_Run2_NonTrig_Tight') and \
            self.testVertex(electron)

    def testLeg2Iso(self, leg, isocut):  # electron
        if isocut is None:
            isocut = self.cfg_ana.iso2
        return leg.relIso(dBetaFactor=0.5, allCharged=0) < isocut

    def testLooseLeg2(self, leg):  # electrons
        ''' pt, eta and isolation selection for electrons
            used in the di-electron veto.
            POG_PHYS14_25ns_v1_Veto
            pt 15, eta 2.5, dB relIso 0.3
        '''
        if (leg.relIso(dBetaFactor=0.5, allCharged=0) > 0.3 or
            abs(leg.eta()) > 2.5 or
            leg.pt() < 15 or
            leg.cutBasedId('POG_PHYS14_25ns_v1_Veto') == False or
            self.testVertex(leg) == False):
            return False
        return True

    def testTightOtherLepton(self, muon):
        '''Tight muon selection, no isolation requirement'''
        return muon.muonID('POG_ID_Medium') and \
            self.testVertex(muon) and \
            abs(muon.eta()) < 2.4 and \
            muon.pt() > 10. and \
            muon.relIso(dBetaFactor=0.5, allCharged=0) < 0.3


    def thirdLeptonVeto(self, leptons, otherLeptons, isoCut=0.3):
        # count electrons (leg 2)
        vLeptons = [electron for electron in leptons if
                      self.testLegKine(electron, ptcut=10, etacut=2.5) and
                      self.testVertex(electron) and
                      electron.electronID('POG_Cuts_ID_PHYS14_25ns_v1_ConvVeto_Veto') and
                      self.testLeg2Iso(electron, isoCut)]

        # count tight muons
        vOtherLeptons = [muon for muon in otherLeptons if
                           self.testTightOtherLepton(muon)]

        if len(vLeptons) + len(vOtherLeptons) > 1:
            return False

        return True

    def leptonAccept(self, leptons):
        '''Returns True if the additional lepton veto is successful'''
        looseLeptons = filter(self.testLooseLeg2, leptons)
        nLeptons = len(looseLeptons)
        if nLeptons < 2:
            return True

        # Reject if OS
        if any(l.charge() > 0 for l in looseLeptons) and \
           any(l.charge() < 0 for l in looseLeptons):
            return False

        return True

    def bestDiLepton(self, diLeptons):
        '''Returns the best diLepton (1st precedence opposite-sign, 2nd precedence
        highest pt1 + pt2).'''

        osDiLeptons = [dl for dl in diLeptons if dl.leg1().charge() != dl.leg2().charge()]
        if osDiLeptons:
            return max(osDiLeptons, key=operator.methodcaller('sumPt'))
        else:
            return max(diLeptons, key=operator.methodcaller('sumPt'))
