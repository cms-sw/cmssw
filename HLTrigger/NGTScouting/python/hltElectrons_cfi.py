import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

##################### Tables for final output and docs ##########################
hltElectronTable = cms.EDProducer("HLTElectronTableProducer",
     skipNonExistingSrc = cms.bool(True),
     src = cms.InputTag("hltEgammaPacker"),
     cut = cms.string(""),
     name = cms.string("hltElectron"),
     doc  = cms.string("HLT Electron information"),
     singleton = cms.bool(False),
     extension = cms.bool(False),
     variables = cms.PSet(
         pt = Var('pt', 'float', precision=10, doc='super-cluster (SC) pt'),
         eta = Var('eta', 'float', precision=10, doc='SC eta'),
         phi = Var('phi', 'float', precision=10, doc='SC phi'),
         m = Var('m', 'float', precision=10, doc='SC mass'),
         dEtaIn = Var('dEtaIn', 'float', precision=10, doc='#Delta#eta(SC seed, track pixel seed)'),
         dPhiIn = Var('dPhiIn', 'float', precision=10, doc='#Delta#phi(SC seed, track pixel seed)'),
         sigmaIetaIeta = Var('sigmaIetaIeta', 'float', precision=10, doc='sigmaIetaIeta of the SC, calculated with full 5x5 region, noise cleaned'),
         hOverE = Var('hOverE', 'float', precision=10, doc='Energy in HCAL / Energy in ECAL'),
         ooEMOop = Var('ooEMOop', 'float', precision=10, doc='1/E(SC) - 1/p(track momentum)'),
         missingHits = Var('missingHits', 'int', doc='missing hits in the tracker'),
         ecalIso = Var('ecalIso', 'float', precision=10, doc='Isolation of SC in the ECAL'),
         hcalIso = Var('hcalIso', 'float', precision=10, doc='Isolation of SC in the HCAL'),
         trackIso = Var('trackIso', 'float', precision=10, doc='Isolation of electron track in the tracker'),
         r9 = Var('r9', 'float', precision=10, doc='ELectron SC r9 as defined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEgammaShowerShape'),
         sMin = Var('sMin', 'float', precision=10, doc='minor moment of the SC shower shape'),
         sMaj = Var('sMaj', 'float', precision=10, doc='major moment of the SC shower shape'),
         seedId = Var('seedId', 'int', doc='ECAL ID of the SC seed'),
     )
)
