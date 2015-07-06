import operator

from PhysicsTools.Heppy.analyzers.core.AutoHandle       import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects   import Muon, GenParticle
# RIC: 16/2/15 need to fix the Electron object first
# from PhysicsTools.Heppy.physicsobjects.HTauTauElectron  import HTauTauElectron as Electron
from PhysicsTools.Heppy.physicsobjects.Electron         import Electron

from CMGTools.H2TauTau.proto.analyzers.DiLeptonAnalyzer import DiLeptonAnalyzer
from CMGTools.H2TauTau.proto.physicsobjects.DiObject    import MuonElectron

class MuEleAnalyzer( DiLeptonAnalyzer ):

    DiObjectClass    = MuonElectron
    LeptonClass      = Muon
    OtherLeptonClass = Electron

    def declareHandles(self):
        super(MuEleAnalyzer, self).declareHandles()
        self.handles  ['diLeptons'   ] = AutoHandle('cmgMuEleCorSVFitFullSel', 'std::vector<pat::CompositeCandidate>')
        self.handles  ['otherLeptons'] = AutoHandle('slimmedElectrons'       , 'std::vector<pat::Electron>'          )
        self.handles  ['leptons'     ] = AutoHandle('slimmedMuons'           , 'std::vector<pat::Muon>'              )
        self.mchandles['genParticles'] = AutoHandle('prunedGenParticles'     , 'std::vector<reco::GenParticle>'      )

    def buildDiLeptons(self, cmgDiLeptons, event):
        '''Build di-leptons, associate best vertex to both legs,
        select di-leptons with a tight ID muon.
        The tight ID selection is done so that dxy and dz can be computed
        (the muon must not be standalone).
        '''
        diLeptons = []
        for index, dil in enumerate(cmgDiLeptons):
            pydil = self.__class__.DiObjectClass(dil)
            # pydil = MuonElectron(dil)
            pydil.leg1().associatedVertex = event.goodVertices[0]
            pydil.leg2().associatedVertex = event.goodVertices[0]
            pydil.leg2().rho = event.rho
            if not self.testLeg2( pydil.leg2(), 999999 ):
                continue
            # pydil.mvaMetSig = pydil.met().getSignificanceMatrix()
            diLeptons.append( pydil )
            pydil.mvaMetSig = pydil.met().getSignificanceMatrix()
        return diLeptons

    def buildLeptons(self, cmgLeptons, event):
        '''Build muons for veto, associate best vertex, select loose ID muons.
        The loose ID selection is done to ensure that the muon has an inner track.'''
        leptons = []
        for index, lep in enumerate(cmgLeptons):
            pyl = self.__class__.LeptonClass(lep)
            #pyl = Muon(lep)
            pyl.associatedVertex = event.goodVertices[0]
            leptons.append( pyl )
        return leptons

    def buildOtherLeptons(self, cmgOtherLeptons, event):
        '''Build electrons for third lepton veto, associate best vertex.
        '''
        otherLeptons = []
        for index, lep in enumerate(cmgOtherLeptons):
            pyl = self.__class__.OtherLeptonClass(lep)
            #import pdb ; pdb.set_trace()
            #pyl = Electron(lep)
            pyl.associatedVertex = event.goodVertices[0]
            pyl.rho = event.rho
            otherLeptons.append( pyl )
        return otherLeptons

    def process(self, event):

        result = super(MuEleAnalyzer, self).process(event)

        if result is False:
            # trying to get a dilepton from the control region.
            # it must have well id'ed and trig matched legs,
            # di-lepton and tri-lepton veto must pass
            result = self.selectionSequence(event, fillCounter = False,
                                            leg1IsoCut = self.cfg_ana.looseiso1,
                                            leg2IsoCut = self.cfg_ana.looseiso2)
            if result is False:
                # really no way to find a suitable di-lepton,
                # even in the control region
                return False
            event.isSignal = False
        else:
            event.isSignal = True

        event.genMatched = None
        if self.cfg_comp.isMC:
            # print event.eventId
            genParticles = self.mchandles['genParticles'].product()
            event.genParticles = map( GenParticle, genParticles)
            leg1DeltaR, leg2DeltaR = event.diLepton.match( event.genParticles )
            if leg1DeltaR>-1 and leg1DeltaR < 0.1 and \
               leg2DeltaR>-1 and leg2DeltaR < 0.1:
                event.genMatched = True
            else:
                event.genMatched = False

        return True

    def testLeg1ID(self, muon):
        '''Tight muon selection, no isolation requirement'''
        # RIC: 9 March 2015
        return muon.muonID('POG_ID_Medium')

    def testLeg1Iso(self, muon, isocut):
        '''Muon isolation to be implemented'''
        # RIC: this leg is the muon,
        # needs to be implemented here
        # For now taken straight from mt channel
        if isocut is None:
            isocut = self.cfg_ana.iso1
        return muon.relIso(dBetaFactor=0.5, allCharged=0)<isocut

    def testVertex(self, lepton):
        '''Tests vertex constraints, for mu and electron'''
        return abs(lepton.dxy()) < 0.045 and abs(lepton.dz ()) < 0.2

    def testLeg2ID(self, electron):
        '''Electron ID. To be implemented'''
        # RIC: this leg is the electron,
        # needs to be implemented here
        # For now taken straight from et channel
        return electron.electronID('POG_MVA_ID_Run2_NonTrig_Tight') and \
            self.testVertex(electron)

    def testLeg2Iso(self, electron, isocut):
        '''Electron Isolation. Relative isolation
           dB corrected factor 0.5
           all charged aprticles
        '''
        # RIC: this leg is the electron,
        # needs to be implemented here
        # For now taken straight from et channel
        if isocut is None:
            isocut = self.cfg_ana.iso2
        return electron.relIso(dBetaFactor=0.5, allCharged=0) < isocut

    def thirdLeptonVeto(self, leptons, otherLeptons, ptcut = 10, isocut = 0.3) :
        '''The tri-lepton veto. To be implemented'''
        return True

    def leptonAccept(self, leptons):
        '''The di-lepton veto, returns false if > one lepton.
        e.g. > 1 mu in the mu tau channel.
        To be implemented.'''
        return True

    def bestDiLepton(self, diLeptons):
        '''Returns the best diLepton (1st precedence opposite-sign,
        2nd precedence highest pt1 + pt2).'''
        osDiLeptons = [dl for dl in diLeptons if dl.leg1().charge() != dl.leg2().charge()]
        if osDiLeptons : return max( osDiLeptons, key=operator.methodcaller( 'sumPt' ) )
        else           : return max(   diLeptons, key=operator.methodcaller( 'sumPt' ) )

