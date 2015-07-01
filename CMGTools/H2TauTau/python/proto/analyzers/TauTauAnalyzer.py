from PhysicsTools.HeppyCore.statistics.counter              import Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle           import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects       import Tau, Muon, Jet, GenParticle
from PhysicsTools.Heppy.physicsobjects.Electron             import Electron

from CMGTools.H2TauTau.proto.analyzers.DiLeptonAnalyzer     import DiLeptonAnalyzer
from CMGTools.H2TauTau.proto.physicsobjects.DiObject        import TauTau

class TauTauAnalyzer( DiLeptonAnalyzer ) :

  DiObjectClass    = TauTau
  LeptonClass      = Muon
  OtherLeptonClass = Electron

  def declareHandles(self):
    super(TauTauAnalyzer, self).declareHandles()
    self.handles['diLeptons'    ] = AutoHandle( 'cmgDiTauCorSVFitFullSel', 'std::vector<pat::CompositeCandidate>' )
    self.handles['otherLeptons' ] = AutoHandle( 'slimmedElectrons'       , 'std::vector<pat::Electron>'           )
    self.handles['leptons'      ] = AutoHandle( 'slimmedMuons'           , 'std::vector<pat::Muon>'               )
    self.handles['jets'         ] = AutoHandle( 'slimmedJets'            , 'std::vector<pat::Jet>'                )

  def process(self, event):

    # method inherited from parent class DiLeptonAnalyzer
    # asks for at least on di-tau pair
    # applies the third lepton veto
    # tests leg1 and leg2
    # cleans by dR the two signal leptons
    # applies the trigger matching to the two signal leptons
    # choses the best di-tau pair, with the bestDiLepton method
    # as implemented here

    result = super(TauTauAnalyzer, self).process(event)

    if result :
      event.isSignal = True
    else :
      # trying to get a dilepton from the control region.
      # it must have well id'ed and trig matched legs,
      # di-lepton and tri-lepton veto must pass
      result = self.selectionSequence( event,
                                       fillCounter = False                  ,
                                       leg1IsoCut  = self.cfg_ana.looseiso1 ,
                                       leg2IsoCut  = self.cfg_ana.looseiso2 )

      if result is False:
          # really no way to find a suitable di-lepton,
          # even in the control region
          return False
      event.isSignal = False

    if not ( hasattr(event, 'leg1') and hasattr(event, 'leg2') ) :
      return False

    # make sure that the legs are sorted by pt
    if event.leg1.pt() < event.leg2.pt() :
      event.leg1 = event.diLepton.leg2()
      event.leg2 = event.diLepton.leg1()
      event.selectedLeptons = [event.leg2, event.leg1]

    return True

  def buildDiLeptons(self, cmgDiLeptons, event):
    '''Build di-leptons, associate best vertex to both legs.'''
    diLeptons = []
    for index, dil in enumerate(cmgDiLeptons):
      pydil = TauTau(dil)
      pydil.leg1().associatedVertex = event.goodVertices[0]
      pydil.leg2().associatedVertex = event.goodVertices[0]
      diLeptons.append( pydil )
      pydil.mvaMetSig = pydil.met().getSignificanceMatrix()
    return diLeptons

  def buildLeptons(self, cmgLeptons, event):
    '''Build muons for veto, associate best vertex, select loose ID muons.
    The loose ID selection is done to ensure that the muon has an inner track.'''
    leptons = []
    for index, lep in enumerate(cmgLeptons):
      pyl = Muon(lep)
      pyl.associatedVertex = event.goodVertices[0]
      if not pyl.muonID('POG_ID_Medium')                     : continue
      if not pyl.relIso(dBetaFactor=0.5, allCharged=0) < 0.3 : continue
      if not self.testLegKine(pyl, ptcut=10, etacut=2.4)     : continue
      leptons.append( pyl )
    return leptons

  def buildOtherLeptons(self, cmgOtherLeptons, event):
    '''Build electrons for third lepton veto, associate best vertex.'''
    otherLeptons = []
    for index, lep in enumerate(cmgOtherLeptons):
      pyl = Electron(lep)
      pyl.associatedVertex = event.goodVertices[0]
      pyl.rho = event.rho
      if not pyl.cutBasedId('POG_PHYS14_25ns_v1_Veto')       : continue
      if not pyl.relIso(dBetaFactor=0.5, allCharged=0) < 0.3 : continue
      if not self.testLegKine(pyl, ptcut=10, etacut=2.5)     : continue
      otherLeptons.append( pyl )
    return otherLeptons

  def testLeg(self, leg, leg_pt, leg_eta, iso, isocut):
    '''requires loose isolation, pt, eta and minimal tauID cuts'''
    return ( self.testTauVertex(leg)                          and
             leg.tauID(iso)                         < isocut  and
             leg.pt()                               > leg_pt  and
             abs(leg.eta())                         < leg_eta and
             (leg.tauID('decayModeFinding')         > 0.5     or
              leg.tauID('decayModeFindingNewDMs')   > 0.5)    and
             leg.tauID('againstElectronVLooseMVA5') > 0.5     and
             leg.tauID('againstMuonLoose3')         > 0.5       )

  def testLeg1(self, leg, isocut):
    leg_pt  = self.cfg_ana.pt1
    leg_eta = self.cfg_ana.eta1
    iso     = self.cfg_ana.isolation
    return self.testLeg(leg, leg_pt, leg_eta, iso, isocut)

  def testLeg2(self, leg, isocut):
    leg_pt  = self.cfg_ana.pt2
    leg_eta = self.cfg_ana.eta2
    iso     = self.cfg_ana.isolation
    return self.testLeg(leg, leg_pt, leg_eta, iso, isocut)

  def testTauVertex(self, lepton):
    '''Tests vertex constraints, for tau'''
    isPV = lepton.vertex().z() == lepton.associatedVertex.z()
    return isPV

  def testVertex(self, lepton, dxy = 0.045, dz = 0.2):
    '''Tests vertex constraints, for mu, e and tau'''
    return abs(lepton.dxy()) < dxy and \
           abs(lepton.dz())  < dz

  def bestDiLepton(self, diLeptons):
    '''Returns the best diLepton (1st precedence most isolated opposite-sign,
    2nd precedence most isolated).'''
    osDiLeptons = [dl for dl in diLeptons if dl.leg1().charge() != dl.leg2().charge()]
    least_iso = lambda dl : max(dl.leg1().tauID(self.cfg_ana.isolation), dl.leg2().tauID(self.cfg_ana.isolation))
    # set reverse = True in case the isolation changes to MVA
    # in that case the least isolated is the one with the lowest MVAscore
    if osDiLeptons : return sorted(osDiLeptons, key=lambda dl : least_iso(dl), reverse=False)[0]
    else           : return sorted(  diLeptons, key=lambda dl : least_iso(dl), reverse=False)[0]
