import FWCore.ParameterSet.Config as cms

# values tuned also according to slide 3 of :
# https://indico.cern.ch/getFile.py/access?contribId=23&sessionId=2&resId=0&materialId=slides&confId=271548
# selection efficiency of approx 6% for ZMM_8TeV

myZgammaFilter = cms.EDFilter('ZgammaMassFilter',

  HepMCProduct             = cms.string("generator"),

  minPhotonPt              = cms.double(7.),
  minLeptonPt              = cms.double(7.),

  minPhotonEta             = cms.double(-3),
  minLeptonEta             = cms.double(-3),

  maxPhotonEta             = cms.double(3),
  maxLeptonEta             = cms.double(3),

  minDileptonMass          = cms.double(30.),
  minZgMass                = cms.double(40.)
 )

ZgammaFilter = cms.Sequence( myZgammaFilter )
