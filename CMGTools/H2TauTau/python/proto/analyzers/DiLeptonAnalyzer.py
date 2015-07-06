import operator

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Lepton
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaR2

from CMGTools.H2TauTau.proto.physicsobjects.DiObject import DiObject


class DiLeptonAnalyzer(Analyzer):

    """Generic analyzer for Di-Leptons.

    Originally in RootTools, then under heppy examples,
    copied from there to not rely on an example.

    Example configuration, and list of parameters:
    #O means optional

    ana = cfg.Analyzer(
        DiLeptonAnalyzer,
        'DiLeptonAnalyzer',
        pt1=20, # pt, eta, iso cuts for leg 1
        eta1=2.3,
        iso1=None,
        pt2=20, # same for leg 2
        eta2=2.1,
        iso2=0.1,
        m_min=10, # mass range
        m_max=99999,
        dR_min=0.5, #O min delta R between the two legs
        allTriggerObjMatched=False,
        verbose=False #from base Analyzer class
        )
    """

    # The DiObject class will be used as the di-object class
    # and the Lepton class as the lepton class
    # Child classes override this choice, and can e.g. decide to use
    # the TauMuon class as a di-object class
    DiObjectClass = DiObject
    LeptonClass = Lepton
    OtherLeptonClass = Lepton

    def beginLoop(self, setup):
        super(DiLeptonAnalyzer, self).beginLoop(setup)
        self.counters.addCounter('DiLepton')
        count = self.counters.counter('DiLepton')
        count.register('all events')
        count.register('> 0 di-lepton')
        count.register('lepton accept')
        count.register('third lepton veto')
        count.register('leg1 offline cuts passed')
        count.register('leg2 offline cuts passed')
        count.register('trig matched')
        count.register('{min:3.1f} < m < {max:3.1f}'.format(min=self.cfg_ana.m_min,
                                                            max=self.cfg_ana.m_max))
        if hasattr(self.cfg_ana, 'dR_min'):
            count.register('dR > {min:3.1f}'.format(min=self.cfg_ana.dR_min))

        count.register('exactly 1 di-lepton')

    def buildDiLeptons(self, cmgDiLeptons, event):
        '''Creates python DiLeptons from the di-leptons read from the disk.
        to be overloaded if needed.'''
        return map(self.__class__.DiObjectClass, cmgDiLeptons)

    def buildLeptons(self, cmgLeptons, event):
        '''Creates python Leptons from the leptons read from the disk.
        to be overloaded if needed.'''
        return map(self.__class__.LeptonClass, cmgLeptons)

    def buildOtherLeptons(self, cmgLeptons, event):
        '''Creates python Leptons from the leptons read from the disk.
        to be overloaded if needed.'''
        return map(self.__class__.LeptonClass, cmgLeptons)

    def process(self, event):
        self.readCollections(event.input)

        event.diLeptons = self.buildDiLeptons(
            self.handles['diLeptons'].product(), event)
        event.leptons = self.buildLeptons(
            self.handles['leptons'].product(), event)
        event.otherLeptons = self.buildOtherLeptons(
            self.handles['otherLeptons'].product(), event)
        return self.selectionSequence(event, fillCounter=True,
                                      leg1IsoCut=self.cfg_ana.iso1,
                                      leg2IsoCut=self.cfg_ana.iso2)

    def selectionSequence(self, event, fillCounter, leg1IsoCut=None, leg2IsoCut=None):

        if fillCounter:
            self.counters.counter('DiLepton').inc('all events')

        if len(event.diLeptons) == 0:
            return False

        if fillCounter:
            self.counters.counter('DiLepton').inc('> 0 di-lepton')

        # testing di-lepton itself
        selDiLeptons = event.diLeptons

        event.leptonAccept = False
        if self.leptonAccept(event.leptons):
            if fillCounter:
                self.counters.counter('DiLepton').inc('lepton accept')
            event.leptonAccept = True

        event.thirdLeptonVeto = False
        if self.thirdLeptonVeto(event.leptons, event.otherLeptons):
            if fillCounter:
                self.counters.counter('DiLepton').inc('third lepton veto')
            event.thirdLeptonVeto = True

        # testing leg1
        selDiLeptons = [diL for diL in selDiLeptons if
                        self.testLeg1(diL.leg1(), leg1IsoCut)]

        if len(selDiLeptons) == 0:
            return False
        elif fillCounter:
            self.counters.counter('DiLepton').inc('leg1 offline cuts passed')

        # testing leg2
        selDiLeptons = [diL for diL in selDiLeptons if
                        self.testLeg2(diL.leg2(), leg2IsoCut)]
        if len(selDiLeptons) == 0:
            return False
        else:
            if fillCounter:
                self.counters.counter('DiLepton').inc(
                    'leg2 offline cuts passed')

        # Trigger matching; both legs
        if len(self.cfg_comp.triggers) > 0:
            requireAllMatched = hasattr(self.cfg_ana, 'allTriggerObjMatched') \
                and self.cfg_ana.allTriggerObjMatched
            selDiLeptons = [diL for diL in selDiLeptons if
                            self.trigMatched(event, diL, requireAllMatched)]

            if len(selDiLeptons) == 0:
                return False
            elif fillCounter:
                self.counters.counter('DiLepton').inc('trig matched')

        # mass cut
        selDiLeptons = [diL for diL in selDiLeptons if
                        self.testMass(diL)]
        if len(selDiLeptons) == 0:
            return False
        else:
            if fillCounter:
                self.counters.counter('DiLepton').inc(
                    '{min:3.1f} < m < {max:3.1f}'.format(min=self.cfg_ana.m_min,
                                                         max=self.cfg_ana.m_max)
                )

        # delta R cut
        if hasattr(self.cfg_ana, 'dR_min'):
            selDiLeptons = [diL for diL in selDiLeptons if
                            self.testDeltaR(diL)]
            if len(selDiLeptons) == 0:
                return False
            else:
                if fillCounter:
                    self.counters.counter('DiLepton').inc(
                        'dR > {min:3.1f}'.format(min=self.cfg_ana.dR_min)
                    )

        # exactly one?
        if len(selDiLeptons) == 0:
            return False
        elif len(selDiLeptons) == 1:
            if fillCounter:
                self.counters.counter('DiLepton').inc('exactly 1 di-lepton')

        event.diLepton = self.bestDiLepton(selDiLeptons)
        event.leg1 = event.diLepton.leg1()
        event.leg2 = event.diLepton.leg2()
        event.selectedLeptons = [event.leg1, event.leg2]

        return True

    def declareHandles(self):
        super(DiLeptonAnalyzer, self).declareHandles()

    def leptonAccept(self, leptons):
        '''Should implement a default version running on event.leptons.'''
        return True

    def thirdLeptonVeto(self, leptons, otherLeptons, isoCut=0.3):
        '''Should implement a default version running on event.leptons.'''
        return True

    def testLeg1(self, leg, isocut=None):
        '''returns testLeg1ID && testLeg1Iso && testLegKine for leg1'''
        return self.testLeg1ID(leg) and \
            self.testLeg1Iso(leg, isocut) and \
            self.testLegKine(leg, self.cfg_ana.pt1, self.cfg_ana.eta1)

    def testLeg2(self, leg, isocut=None):
        '''returns testLeg2ID && testLeg2Iso && testLegKine for leg2'''
        return self.testLeg2ID(leg) and \
            self.testLeg2Iso(leg, isocut) and \
            self.testLegKine(leg, self.cfg_ana.pt2, self.cfg_ana.eta2)

    def testLegKine(self, leg, ptcut, etacut):
        '''Tests pt and eta.'''
        return leg.pt() > ptcut and \
            abs(leg.eta()) < etacut

    def testLeg1ID(self, leg):
        '''Always return true by default, overload in your subclass'''
        return True

    def testLeg1Iso(self, leg, isocut):
        '''If isocut is None, the iso value is taken from the iso1 parameter.
        Checks the standard dbeta corrected isolation.
        '''
        if isocut is None:
            isocut = self.cfg_ana.iso1
        return leg.relIso(0.5) < isocut

    def testLeg2ID(self, leg):
        '''Always return true by default, overload in your subclass'''
        return True

    def testLeg2Iso(self, leg, isocut):
        '''If isocut is None, the iso value is taken from the iso2 parameter.
        Checks the standard dbeta corrected isolation.
        '''
        if isocut is None:
            isocut = self.cfg_ana.iso2
        return leg.relIso(0.5) < isocut

    def testMass(self, diLepton):
        '''returns True if the mass of the dilepton is between the m_min and m_max parameters'''
        mass = diLepton.mass()
        return self.cfg_ana.m_min < mass and mass < self.cfg_ana.m_max

    def testDeltaR(self, diLepton):
        '''returns True if the two diLepton.leg1() and .leg2() have a delta R larger than the dR_min parameter.'''
        dR = deltaR(diLepton.leg1().eta(), diLepton.leg1().phi(),
                    diLepton.leg2().eta(), diLepton.leg2().phi())
        return dR > self.cfg_ana.dR_min

    def bestDiLepton(self, diLeptons):
        '''Returns the best diLepton (the one with highest pt1 + pt2).'''
        return max(diLeptons, key=operator.methodcaller('sumPt'))

    def trigMatched(self, event, diL, requireAllMatched=False):
        '''Check that at least one trigger object per pgdId from a given trigger 
        has a matched leg with the same pdg ID. If requireAllMatched is True, 
        requires that each single trigger object has a match.'''
        matched = False
        legs = [diL.leg1(), diL.leg2()]
        event.matchedPaths = set()

        for info in event.trigger_infos:
            if not info.fired:
                continue

            matchedIds = set()
            allMatched = True
            for to in info.objects:
                if self.trigObjMatched(to, legs):
                    matchedIds.add(abs(to.pdgId()))
                else:
                    allMatched = False

            if matchedIds == info.objIds:
                if requireAllMatched and not allMatched:
                    matched = False
                else:
                    matched = True
                    event.matchedPaths.add(info.name)

        return matched

    def trigObjMatched(self, to, legs, dR2Max=0.25):  # dR2Max=0.089999
        '''Returns true if the trigger object is matched to one of the given
        legs'''
        eta = to.eta()
        phi = to.phi()
        pdgId = abs(to.pdgId())
        to.matched = False
        for leg in legs:
            # JAN - Single-ele trigger filter has pdg ID 0, to be understood
            if pdgId == 0 or pdgId == abs(leg.pdgId()):
                if deltaR2(eta, phi, leg.eta(), leg.phi()) < dR2Max:
                    to.matched = True
                    # leg.trigMatched = True

        return to.matched
