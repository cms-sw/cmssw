from PhysicsTools.Heppy.analyzers.DiLeptonAnalyzer import DiLeptonAnalyzer
from PhysicsTools.Heppy.analyzers.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.DiObject import DiMuon
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Muon


class ZMuMuAnalyzer( DiLeptonAnalyzer ):

    DiObjectClass = DiMuon
    LeptonClass = Muon

    def declareHandles(self):
        super(ZMuMuAnalyzer, self).declareHandles()
        print 'ZMuMuAnalyzer.declareHandles'
        self.handles['diLeptons'] = AutoHandle(
            'cmgDiMuonSel',
            'std::vector<cmg::DiObject<cmg::Muon,cmg::Muon>>'
            )
        self.handles['leptons'] = AutoHandle(
            'cmgMuonSel',
            'std::vector<cmg::Muon>'
            )
        self.handles['otherLeptons'] = AutoHandle(
            'cmgElectronSel',
            'std::vector<cmg::Electron>'
            )


    def buildDiLeptons(self, cmgDiLeptons, event):
        '''Build di-leptons, associate best vertex to both legs,
        select di-leptons with a tight ID muon.
        The tight ID selection is done so that dxy and dz can be computed
        (the muon must not be standalone).
        '''
        diLeptons = []
        for index, dil in enumerate(cmgDiLeptons):
            pydil = self.__class__.DiObjectClass(dil)
            pydil.leg1().associatedVertex = event.goodVertices[0]
            pydil.leg2().associatedVertex = event.goodVertices[0]
            diLeptons.append( pydil )
        return diLeptons


    def buildLeptons(self, cmgLeptons, event):
        return []


    def buildOtherLeptons(self, cmgLeptons, event):
        return []
    

    def testVertex(self, lepton):
        '''Tests vertex constraints, for mu and tau'''
        return abs(lepton.dxy()) < 0.045 and \
               abs(lepton.dz()) < 0.2 


    def testMuonIso(self, muon, isocut ):
        '''dbeta corrected pf isolation with all charged particles instead of
        charged hadrons'''
        return muon.relIsoAllChargedDB05()<isocut

    testLeg1Iso = testMuonIso
    testLeg2Iso = testMuonIso

    def testMuonID(self, muon):
        '''Tight muon selection, no isolation requirement'''
        # import pdb; pdb.set_trace()
        return muon.tightId() and \
               self.testVertex( muon )          


    testLeg1ID = testMuonID
    testLeg2ID = testMuonID
