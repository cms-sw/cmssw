import FWCore.ParameterSet.Config as cms

phfCoincFilter2Th4  = cms.EDFilter('HiHFFilter',
   HFfilters      = cms.InputTag("hiHFfilters","hiHFfilters"),
   threshold      = cms.int32(4),
   minnumtowers  = cms.int32(2)
)

phfCoincFilter1Th4 = phfCoincFilter2Th4.clone(minnumtowers = 1)
phfCoincFilter3Th4 = phfCoincFilter2Th4.clone(minnumtowers = 3)
phfCoincFilter4Th4 = phfCoincFilter2Th4.clone(minnumtowers = 4)
phfCoincFilter5Th4 = phfCoincFilter2Th4.clone(minnumtowers = 5)

phfCoincFilter1Th3 = phfCoincFilter2Th4.clone(threshold = 3, minnumtowers = 1)
phfCoincFilter2Th3 = phfCoincFilter2Th4.clone(threshold = 3, minnumtowers = 2)
phfCoincFilter3Th3 = phfCoincFilter2Th4.clone(threshold = 3, minnumtowers = 3)
phfCoincFilter4Th3 = phfCoincFilter2Th4.clone(threshold = 3, minnumtowers = 4)
phfCoincFilter5Th3 = phfCoincFilter2Th4.clone(threshold = 3, minnumtowers = 5)

phfCoincFilter1Th5 = phfCoincFilter2Th4.clone(threshold = 5, minnumtowers = 1)
phfCoincFilter2Th5 = phfCoincFilter2Th4.clone(threshold = 5, minnumtowers = 2)
phfCoincFilter3Th5 = phfCoincFilter2Th4.clone(threshold = 5, minnumtowers = 3)
phfCoincFilter4Th5 = phfCoincFilter2Th4.clone(threshold = 5, minnumtowers = 4)
phfCoincFilter5Th5 = phfCoincFilter2Th4.clone(threshold = 5, minnumtowers = 5)

phfCoincFilter4Th2 = phfCoincFilter2Th4.clone(threshold = 2, minnumtowers = 4)

pphfCoincFilter4Th2 = cms.Path(phfCoincFilter4Th2)
pphfCoincFilter1Th3 = cms.Path(phfCoincFilter1Th3)
pphfCoincFilter2Th3 = cms.Path(phfCoincFilter2Th3)
pphfCoincFilter3Th3 = cms.Path(phfCoincFilter3Th3)
pphfCoincFilter4Th3 = cms.Path(phfCoincFilter4Th3)
pphfCoincFilter5Th3 = cms.Path(phfCoincFilter5Th3)
pphfCoincFilter1Th4 = cms.Path(phfCoincFilter1Th4)
pphfCoincFilter2Th4 = cms.Path(phfCoincFilter2Th4)
pphfCoincFilter3Th4 = cms.Path(phfCoincFilter3Th4)
pphfCoincFilter4Th4 = cms.Path(phfCoincFilter4Th4)
pphfCoincFilter5Th4 = cms.Path(phfCoincFilter5Th4)
pphfCoincFilter1Th5 = cms.Path(phfCoincFilter1Th5)
pphfCoincFilter2Th5 = cms.Path(phfCoincFilter2Th5)
pphfCoincFilter3Th5 = cms.Path(phfCoincFilter3Th5)
pphfCoincFilter4Th5 = cms.Path(phfCoincFilter4Th5)
pphfCoincFilter5Th5 = cms.Path(phfCoincFilter5Th5)
