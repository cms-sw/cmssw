import FWCore.ParameterSet.Config as cms

# Associate SiStrip electrons to tracks
# $ Id: $
# Author: Jim Pivarski, Cornell 3 Aug 2006
#
siStripElectronToTrackAssociator = cms.EDFilter("SiStripElectronAssociator",
    siStripElectronCollection = cms.string(''),
    trackCollection = cms.string(''),
    electronsLabel = cms.string('siStripElectrons'),
    siStripElectronProducer = cms.string('siStripElectrons'),
    trackProducer = cms.string('egammaCTFFinalFitWithMaterial')
)


