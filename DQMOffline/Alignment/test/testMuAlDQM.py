import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonAlignmentMonitor")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000000

process.load("DQMOffline.Alignment.muonAlignment_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")


# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")

# reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

#from DQMOffline.Alignment.input_cfi import source

#process.source = source

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V4_StreamALCARECOMuAlCalIsolatedMu_step2_AlcaReco-v1/0008/001A49E8-93C3-DD11-9720-003048D15CFA.root')
#fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOMuAlStandAloneCosmics_225-v3/0008/4EE39297-01FF-DD11-82CF-003048678C9A.root')
fileNames= cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_227_Tosca090216_ReReco_FromTrackerPointing_v2/0006/F453F276-8421-DE11-ABFE-001731AF66A7.root')
)

#process.muonAlignment.MuonCollection = "cosmicMuons"
process.muonAlignment.MuonCollection = "ALCARECOMuAlStandAloneCosmics"
#process.muonAlignment.MuonCollection = "ALCARECOMuAlCalIsolatedMu:StandAlone"
#process.muonAlignment.MuonCollection = "ALCARECOMuAlCalIsolatedMu:SelectedMuons"
process.muonAlignment.OutputMEsInRootFile = cms.bool(True)
process.muonAlignment.doSummary= cms.untracked.bool(True)


process.myDQM = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_MuonAlignmentMonitor'),
         SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')),
    fileName = cms.untracked.string('dqm.root')
)

#process.p = cms.Path(process.muonAlignment*process.MEtoEDMConverter)

#process.load("DQMServices.Core.DQM_cfg")

#process.dqmSaverMy = cms.EDFilter("DQMFileSaver",
#         convention=cms.untracked.string("Offline"),
#         workflow=cms.untracked.string("/Alignment/Muon/ALCARECOMuAlGlobalCosmics_v11_SAalgo"),
#         dirName=cms.untracked.string("."),
#         saveAtJobEnd=cms.untracked.bool(True),                        
#         forceRunNumber=cms.untracked.int32(1)
#	)


process.myTracks = cms.EDFilter("MyTrackSelector",
#    src = cms.InputTag("ALCARECOMuAlCalIsolatedMu:StandAlone"),
   src = cms.InputTag("ALCARECOMuAlStandAloneCosmics"),
#   src = cms.InputTag("cosmicMuons"),
    cut = cms.string("pt > 40"),
    filter = cms.bool(True)
  )


process.p = cms.Path(
#process.myTracks *
process.muonAlignment
#	*process.dqmSaverMy
#	*process.MEtoEDMConverter
	)

process.outpath = cms.EndPath(process.myDQM)

from CondCore.DBCommon.CondDBSetup_cfi import *

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
if os.environ["CMS_PATH"] != "":
  del process.es_prefer_GlobalTag
  del process.SiStripPedestalsFakeESSource
  del process.siStripBadChannelFakeESSource
  del process.siStripBadFiberFakeESSource
  del process.DTFakeVDriftESProducer


#process.prefer("GlobalTag")
#process.prefer("GlobalTag")
process.GlobalTag.globaltag = 'CRAFT_ALL_V11::All'

process.myAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
#connect = cms.string('sqlite_file:/afs/cern.ch/user/p/pablom/public/DBs/Alignments_CRAFT_ALL_V4_refitter.db'),
#connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/HWAlignment/AlignmentDB/AlignmentsNewEndcap.db'),
connect = cms.string('sqlite_file:/afs/cern.ch/user/p/pivarski/public/CRAFTalignment4_NewTracker_xyphiz2_alignedonlyAPEs.db'),
#DBParameters = CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup.DBParameters,

toGet = cms.VPSet(
  cms.PSet(
    record = cms.string('DTAlignmentRcd'),
    tag = cms.string('DTAlignmentRcd')
  ),
  cms.PSet(
    record = cms.string('DTAlignmentErrorRcd'),
    tag = cms.string('DTAlignmentErrorRcd')
  ),
  cms.PSet(
    record = cms.string('CSCAlignmentRcd'),
    tag = cms.string('CSCAlignmentRcd')
  ),
   cms.PSet(
    record = cms.string('CSCAlignmentErrorRcd'),
    tag = cms.string('CSCAlignmentErrorRcd')
  )
 )
) 
process.es_prefer_myAlignment = cms.ESPrefer("PoolDBESSource","myAlignment")
 


#process.schedule = cms.Schedule(process.p,process.outpath)


