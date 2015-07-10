import operator

from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Muon
from PhysicsTools.Heppy.physicsobjects.Electron import Electron

from CMGTools.H2TauTau.proto.analyzers.DiLeptonAnalyzer import DiLeptonAnalyzer
from CMGTools.H2TauTau.proto.physicsobjects.DiObject import DiMuon

class MuMuAnalyzer(DiLeptonAnalyzer):

    DiObjectClass = DiMuon
    LeptonClass = Muon
    OtherLeptonClass = Electron

    def declareHandles(self):
        super(MuMuAnalyzer, self).declareHandles()
        self.handles['diLeptons'] = AutoHandle(
            'cmgDiMuCorSVFitFullSel',
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
        result = super(MuMuAnalyzer, self).process(event)

        if result is False:
            # trying to get a dilepton from the control region.
            # it must have well id'ed and trig matched legs,
            # di-lepton and tri-lepton veto must pass
            result = self.selectionSequence(event, fillCounter=False,
                                            leg1IsoCut=-9999,
                                            leg2IsoCut=9999)
            if result is False:
                # really no way to find a suitable di-lepton,
                # even in the control region
                return False
            event.isSignal = False
        else:
            event.isSignal = True
      
        return True
        

    def testLeg1ID(self, muon):
        return self.testLeg2ID(muon)
        

    def testLeg1Iso(self, muon, isocut):
        return self.testLeg2Iso(muon, isocut)


    def testVertex(self, lepton):
        '''Tests vertex constraints, for mu'''
        return abs(lepton.dxy()) < 0.045 and abs(lepton.dz()) < 0.2 


    def testLeg2ID(self, muon):
        '''Tight muon selection, no isolation requirement'''
        return muon.tightId() and self.testVertex( muon )
               

    def testLeg2Iso(self, muon, isocut):
        '''Tight muon selection, with isolation requirement'''
        if isocut is None:
            isocut = self.cfg_ana.iso2

        return muon.relIso(dBetaFactor=0.5, allCharged=1) < isocut    


    def thirdLeptonVeto(self, leptons, otherLeptons, ptcut=10, isocut=0.3):
        '''Tri-lepton veto. Returns False if > 2 leptons (e or mu).'''
        vleptons = [lep for lep in leptons if
                    self.testLegKine(lep, ptcut=ptcut, etacut=2.4) and 
                    self.testLeg2ID(lep) and
                    self.testLeg2Iso(lep, isocut)
                   ]
        # count electrons
        votherLeptons = [olep for olep in otherLeptons if 
                         self.testLegKine(olep, ptcut=ptcut, etacut=2.5) and
                         # Take loose MVA ID for now
                         olep.mvaIDLoose() and
                         self.testVertex(olep) and
                         olep.relIso(dBetaFactor=0.5, allCharged=0) < isocut
                        ]
        if len(vleptons) + len(votherLeptons) > 2:
            return False
        
        return True
        

    def leptonAccept(self, leptons):
        '''Di-lepton veto: none applied for now'''
        return True

    def bestDiLepton(self, diLeptons):
        '''Returns the best diLepton (1st precedence opposite-sign, 2nd precedence
        highest pt1 + pt2).'''

        osDiLeptons = [dl for dl in diLeptons if dl.leg1().charge() != dl.leg2().charge()]
        if osDiLeptons:
            return max(osDiLeptons, key=operator.methodcaller('sumPt'))
        else:
            return max(diLeptons, key=operator.methodcaller('sumPt'))

