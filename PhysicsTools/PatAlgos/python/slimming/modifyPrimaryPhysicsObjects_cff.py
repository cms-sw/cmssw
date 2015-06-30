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

modifyPrimaryPhysicsObjects = cms.Sequence( electronMVAValueMapProducer * slimmedElectrons *
                                            photonIDValueMapProducer * photonMVAValueMapProducer * slimmedPhotons *
                                            slimmedMuons     *
                                            slimmedTaus      *
                                            slimmedJets      *
                                            slimmedJetsAK8   *
                                            slimmedJetsPuppi   )
