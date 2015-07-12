import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.modifiedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedPhotons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedMuons_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedTaus_cfi import *
from PhysicsTools.PatAlgos.slimming.modifiedJets_cfi import *

#get any prereqs from POG areas
from RecoEgamma.ElectronIdentification.ElectronMVAValueMapProducer_cfi import *
from RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi import *
from RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi import *

#clone modules so we have slimmed* -> slimmed*
slimmedElectrons = modifiedElectrons.clone()
slimmedPhotons = modifiedPhotons.clone()
slimmedMuons = modifiedMuons.clone()
slimmedTaus = modifiedTaus.clone()
slimmedJets = modifiedJets.clone()
slimmedJetsAK8 = modifiedJets.clone( src = cms.InputTag("slimmedJetsAK8",processName=cms.InputTag.skipCurrentProcess()) )
slimmedJetsPuppi = modifiedJets.clone( src = cms.InputTag("slimmedJetsPuppi",processName=cms.InputTag.skipCurrentProcess()) )

modifyPrimaryPhysicsObjects = cms.Sequence( electronMVAValueMapProducer *
                                            photonIDValueMapProducer * photonMVAValueMapProducer * 
                                            slimmedElectrons *
                                            slimmedPhotons *
                                            slimmedMuons     *
                                            slimmedTaus      *
                                            slimmedJets      *
                                            slimmedJetsAK8   *
                                            slimmedJetsPuppi   )
