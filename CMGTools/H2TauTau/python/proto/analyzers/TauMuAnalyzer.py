import operator

from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
from PhysicsTools.Heppy.physicsobjects.Electron import Electron

from CMGTools.H2TauTau.proto.analyzers.DiLeptonAnalyzer import DiLeptonAnalyzer
from CMGTools.H2TauTau.proto.physicsobjects.DiObject import TauMuon

class TauMuAnalyzer(DiLeptonAnalyzer):

    DiObjectClass = TauMuon
    LeptonClass = Muon
    OtherLeptonClass = Electron

    def declareHandles(self):
        super(TauMuAnalyzer, self).declareHandles()
        self.handles['diLeptons'] = AutoHandle(
            'cmgTauMuCorSVFitFullSel',
            'std::vector<pat::CompositeCandidate>'
            )

        self.handles['otherLeptons'] = AutoHandle(
            'slimmedElectrons',
            'std::vector<pat::Electron>'
            )

        self.handles['leptons'] = AutoHandle(
            'slimmedMuons',
            'std::vector<pat::Muon>'
            )

        self.mchandles['genParticles'] = AutoHandle(
            'prunedGenParticles',
            'std::vector<reco::GenParticle>'
            )


    def buildDiLeptons(self, patDiLeptons, event):
        '''Build di-leptons, associate best vertex to both legs,
        select di-leptons with a tight ID muon.
        The tight ID selection is done so that dxy and dz can be computed
        (the muon must not be standalone).
        '''
        diLeptons = []
        for index, dil in enumerate(patDiLeptons):
            pydil = self.__class__.DiObjectClass(dil)
            pydil.leg1().associatedVertex = event.goodVertices[0]
            pydil.leg2().associatedVertex = event.goodVertices[0]
            if not self.testLeg2(pydil.leg2(), 99999):
                continue
            # JAN: This crashes. Waiting for idea how to fix this; may have
            # to change data format otherwise, though we don't yet strictly
            # need the MET significance matrix here since we calculate SVFit
            # before
            pydil.mvaMetSig = pydil.met().getSignificanceMatrix()
            diLeptons.append(pydil)
        return diLeptons


    def buildLeptons(self, patLeptons, event):
        '''Build muons for veto, associate best vertex, select loose ID muons.
        The loose ID selection is done to ensure that the muon has an inner track.'''
        leptons = []
        for index, lep in enumerate(patLeptons):
            pyl = self.__class__.LeptonClass(lep)
            pyl.associatedVertex = event.goodVertices[0]
            leptons.append(pyl)
        return leptons


    def buildOtherLeptons(self, patOtherLeptons, event):
        '''Build electrons for third lepton veto, associate best vertex.
        '''
        otherLeptons = []
        for index, lep in enumerate(patOtherLeptons):
            pyl = self.__class__.OtherLeptonClass(lep)
            pyl.associatedVertex = event.goodVertices[0]
            pyl.rho = event.rho
            otherLeptons.append(pyl)
        return otherLeptons


    def process(self, event):
        result = super(TauMuAnalyzer, self).process(event)

        if result is False:
            # trying to get a dilepton from the control region.
            # it must have well id'ed and trig matched legs,
            # di-lepton and tri-lepton veto must pass
            result = self.selectionSequence(event, fillCounter=False,
                                            leg1IsoCut=self.cfg_ana.looseiso1,
                                            leg2IsoCut=self.cfg_ana.looseiso2)
            if result is False:
                # really no way to find a suitable di-lepton,
                # even in the control region
                return False
            event.isSignal = False
        else:
            event.isSignal = True

        return True


    def testLeg1ID(self, tau):
        # RIC: 9 March 2015
        return ( (tau.tauID('decayModeFinding')         > 0.5  or
                  tau.tauID('decayModeFindingNewDMs')   > 0.5) and
                 tau.tauID('againstElectronVLooseMVA5') > 0.5  and
                 tau.tauID('againstMuonTight3')         > 0.5  and
                 self.testTauVertex(tau) )
        # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendation13TeV
        # return tau.tauID('decayModeFinding') > 0.5 and \
        #        tau.tauID('againstMuonTight3') > 0.5 and \
        #        tau.tauID('againstElectronLooseMVA5') > 0.5 and \
        #        self.testTauVertex(tau)


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


    def testTauVertex(self, lepton):
        '''Tests vertex constraints, for tau'''
        # Just checks if the primary vertex the tau was reconstructed with
        # corresponds to the one used in the analysis
        isPV = lepton.vertex().z() == lepton.associatedVertex.z()
        return isPV


    def testVertex(self, lepton):
        '''Tests vertex constraints, for mu'''
        return abs(lepton.dxy()) < 0.045 and abs(lepton.dz()) < 0.2


    def testLeg2ID(self, muon):
        '''Tight muon selection, no isolation requirement'''
        return muon.muonID('POG_ID_Medium') and self.testVertex(muon)


    def testLeg2Iso(self, muon, isocut):
        '''Tight muon selection, with isolation requirement'''
        if isocut is None:
            isocut = self.cfg_ana.iso2

        return muon.relIso(dBetaFactor=0.5, allCharged=0) < isocut


    def thirdLeptonVeto(self, leptons, otherLeptons, isoCut=0.3):
        # count electrons (leg 2)
        vOtherLeptons = [electron for electron in otherLeptons if
                           self.testLegKine(electron, ptcut=10, etacut=2.5) and
                           self.testVertex(electron) and
                           electron.cutBasedId('POG_PHYS14_25ns_v1_Veto') and
                           electron.relIso(dBetaFactor=0.5, allCharged=0) < 0.3]

        # count tight muons
        vLeptons = [muon for muon in leptons if
                      muon.muonID('POG_ID_Medium') and
                      self.testVertex(muon) and
                      self.testLegKine(muon, ptcut=10, etacut=2.4) and
                      muon.relIso(dBetaFactor=0.5, allCharged=0) < 0.3]

        if len(vLeptons) + len(vOtherLeptons) > 1:
            return False

        return True



    def leptonAccept(self, leptons):
        '''Di-lepton veto: returns false if >= 1 OS same flavour lepton pair,
        e.g. >= 1 OS mu pair in the mu tau channel'''
        looseLeptons = [muon for muon in leptons if
                        self.testLegKine(muon, ptcut=15, etacut=2.4) and
                        muon.isGlobalMuon() and
                        muon.isTrackerMuon() and
                        muon.userFloat('isPFMuon') and
                        abs(muon.dz()) < 0.2 and
                        self.testLeg2Iso(muon, 0.3)
                       ]

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

