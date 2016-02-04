import FWCore.ParameterSet.Config as cms

# Associate SiStrip electrons to tracks
# $ Id: $
# Author: Jim Pivarski, Cornell 3 Aug 2006
#
siStripElectronToTrackAssociator = cms.EDProducer("SiStripElectronAssociator",
    siStripElectronCollection = cms.InputTag('siStripElectrons'),
    trackCollection = cms.InputTag('egammaCTFFinalFitWithMaterial'),
    electronsLabel = cms.InputTag('siStripElectrons'),
    #siStripElectronProducer = cms.InputTag('siStripElectrons'),
    #trackProducer = cms.InputTag('egammaCTFFinalFitWithMaterial')
)


