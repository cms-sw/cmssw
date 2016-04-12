import FWCore.ParameterSet.Config as cms

# minimum of logs
MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/j/jkaspar/public/run268608_ls0001_streamA_StorageManager.root')
)

# raw to digi conversion
from CondFormats.TotemReadoutObjects.TotemDAQMappingESSourceXML_cfi import *
TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/TotemReadoutObjects/xml/ctpps_210_mapping.xml")

from EventFilter.TotemRawToDigi.TotemRawToDigi_cfi import *
TotemRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")
TotemRawToDigi.fedIds = cms.vuint32(577, 578, 579, 580)
TotemRawToDigi.RawToDigi.printErrorSummary = 0
TotemRawToDigi.RawToDigi.printUnknownFrameSummary = 0

# clusterization
from RecoTotemRP.RPClusterSigmaService.ClusterSigmaServiceConf_cfi import *
from RecoTotemRP.RPClusterizer.RPClusterizationConf_cfi import *
RPClustProd.DigiLabel = cms.InputTag("TotemRawToDigi")

# reco hit production
from RecoTotemRP.RPRecoHitProducer.RPRecoHitProdConf_cfi import *

# TOTEM RP geometry
from Configuration.TotemCommon.geometryRP_cfi import *
XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/TotemRPData/data/RP_Garage/RP_Dist_Beam_Cent.xml")

# non-parallel pattern recognition
from RecoTotemRP.RPNonParallelTrackCandidateFinder.RPNonParallelTrackCandidateFinder_cfi import *
NonParallelTrackFinder.DetSetVectorTotemRPRecHitLabel = cms.InputTag("RPRecoHitProd")
NonParallelTrackFinder.verbosity = 0
NonParallelTrackFinder.maxHitsPerPlaneToSearch = 5
NonParallelTrackFinder.minPlanesPerProjectionToSearch = 2
NonParallelTrackFinder.minPlanesPerProjectionToFit = 3
NonParallelTrackFinder.threshold = 2.99

# local track fitting
from RecoTotemRP.RPTrackCandidateCollectionFitter.RPSingleTrackCandCollFitted_cfi import *
RPSingleTrackCandCollFit.Verbosity = 0
RPSingleTrackCandCollFit.RPTrackCandCollProducer = 'NonParallelTrackFinder'

# processing path
reco_step = cms.Path(
  TotemRawToDigi *
  RPClustProd *
  RPRecoHitProd *
  NonParallelTrackFinder *
  RPSingleTrackCandCollFit
)
