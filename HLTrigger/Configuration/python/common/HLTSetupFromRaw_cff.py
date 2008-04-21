import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.rawToDigi.RawToDigi_cff import *
from L1Trigger.Configuration.L1HltSeed_cff import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from HLTrigger.Configuration.rawToDigi.SiStripRawToClusters_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
from HLTrigger.Configuration.common.EcalRegionalRecoFromRaw_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
doLocalPixel = cms.Sequence(siPixelDigis*siPixelClusters*siPixelRecHits)
doLocalStrip = cms.Sequence(SiStripRawToClusters)
doLocalTracker = cms.Sequence(doLocalPixel+doLocalStrip)
hcalLocalRecoWithoutHO = cms.Sequence(hbhereco+hfreco)
doLocalEcal = cms.Sequence(ecalDigis+ecalPreshowerDigis+ecalLocalRecoSequence)
doLocalEcal_nopreshower = cms.Sequence(ecalDigis+ecalLocalRecoSequence_nopreshower)
doLocalHcal = cms.Sequence(hcalDigis+hcalLocalRecoSequence)
doLocalHcalWithoutHO = cms.Sequence(hcalDigis+hcalLocalRecoWithoutHO)
doLocalCalo = cms.Sequence(doLocalEcal+doLocalHcal)
doLocalCaloWithoutHO = cms.Sequence(doLocalEcal+doLocalHcalWithoutHO)
doRegionalEgammaEcal = cms.Sequence(ecalPreshowerDigis+ecalRegionalEgammaRecoSequence)
doRegionalMuonsEcal = cms.Sequence(ecalPreshowerDigis+ecalRegionalMuonsRecoSequence)
doRegionalTausEcal = cms.Sequence(ecalPreshowerDigis+ecalRegionalTausRecoSequence)
doRegionalJetsEcal = cms.Sequence(ecalPreshowerDigis+ecalRegionalJetsRecoSequence)
doEcalAll = cms.Sequence(ecalPreshowerDigis+ecalAllRecoSequence)
doLocalCSC = cms.Sequence(muonCSCDigis*csclocalreco)
doLocalDT = cms.Sequence(muonDTDigis*dtlocalreco)
doLocalRPC = cms.Sequence(muonRPCDigis*rpcRecHits)
doLocalMuon = cms.Sequence(doLocalDT+doLocalCSC+doLocalRPC)

