import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalAlignment")

### debug
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    statistics = cms.untracked.vstring('alignment'),
#    destinations = cms.untracked.vstring('alignment'),
#    alignment = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#        )
#)

process.load("CondCore.CondDB.CondDB_cfi")

###  number of events ###
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100000))

process.load("Alignment.CommonAlignmentProducer.globalTrackerMuonAlignment_cfi")

### debug  isolated/cosmic  refit
#process.globalTrackerMuonAlignment.debug = True
process.globalTrackerMuonAlignment.cosmics = True
#process.globalTrackerMuonAlignment.isolated = True
process.globalTrackerMuonAlignment.writeDB = True
process.globalTrackerMuonAlignment.refitmuon = True
process.globalTrackerMuonAlignment.refittrack = True

# cosmic muon
process.globalTrackerMuonAlignment.tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics:TrackerOnly")
process.globalTrackerMuonAlignment.muons = cms.InputTag("ALCARECOMuAlGlobalCosmics:StandAlone")
process.globalTrackerMuonAlignment.gmuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")
process.globalTrackerMuonAlignment.smuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:SelectedMuons")
# isolated muon 
#process.globalTrackerMuonAlignment.tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:TrackerOnly")
#process.globalTrackerMuonAlignment.muons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:StandAlone")
#process.globalTrackerMuonAlignment.gmuons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
#process.globalTrackerMuonAlignment.smuons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:SelectedMuons")

process.globalTrackerMuonAlignment.propagator = "SteppingHelixPropagator"
# propagator really not used now
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

process.p = cms.Path(process.globalTrackerMuonAlignment)

#process.dump=cms.EDAnalyzer('EventContentAnalyzer')
#process.p = cms.Path(process.globalTrackerMuonAlignment*process.dump)

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
#from Configuration.StandardSequences.MagneticField_cff import *
### Force Magnetic Field B=3.8T (maybe needed for runs: 12476-127764 of cosmics 2010)
#process.load("Configuration/StandardSequences/MagneticField_38T_cff")

# Geometry
process.load("Configuration.Geometry.GeometryRecoDB_cff")
#from Configuration.StandardSequences.GeometryExtended_cff import *

# Gloabal Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# import of standard configurations Cosmics
process.load('Configuration/EventContent/EventContentCosmics_cff')

### refit
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load('RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi')


#                             GlobalTag
# 3_6_0
#process.GlobalTag.globaltag = 'START36_V2::All'
#process.GlobalTag.globaltag = 'GR09_R_36X_V1::All'
#process.GlobalTag.globaltag = 'MC_36Y_V2::All'
# 3_5_5
#process.GlobalTag.globaltag = 'MC_3XY_V25::All'
# isolated muon 314
#process.GlobalTag.globaltag = "MC_31X_V5::All"
# 314 Cosmics
#process.GlobalTag.globaltag = "COSMMC_22X_V6::All"
# craft08
#process.GlobalTag.globaltag = 'CRAFT0831X_V1::All'
# craft09 small samples
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
# 1st repro craft09
#process.GlobalTag.globaltag = 'CRAFT09_R_V10::All'
# cosmic 2010
process.GlobalTag.globaltag = "GR10_P_V5::All"
#process.GlobalTag.globaltag = 'GR10_P_V2COS::All'
# Pablo
#process.GlobalTag.globaltag = "GR10_P_V4::All"
# Jula
#process.GlobalTag.globaltag = 'GR10_P_V3COS::All'

### write global Rcd to DB
from CondCore.CondDB.CondDB_cfi import *
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    #process.CondDB,
    ### Writing to oracle needs the following shell variable setting (in zsh):
    ### export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    ### string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:output.db'),
    ### untracked uint32 authenticationMethod = 1
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    ))
)
process.CondDB.DBParameters.messageLevel = 2

###  read GlobalPositionRcd
#process.GlobalTag.toGet = cms.VPSet(
#    cms.PSet(
#      record = cms.string("GlobalPositionRcd"),
#      tag = cms.string("IdealGeometry"),
###      connect = cms.untracked.string("sqlite_file:zeroGlobalPosRcd.db")
###      connect = cms.untracked.string("sqlite_file:output_rcd1z.db")
###      connect = cms.untracked.string("sqlite_file:output_rcd03450123.db")
#      connect = cms.untracked.string("sqlite_file:output_rcdzero.db")
#      )
# )

#### read Muon Geometry ###  Pablo cff
#import CondCore.CondDB.CondDB_cfi
#process.muonAlignment = cms.ESSource("PoolDBESSource",
#connect = cms.string('sqlite_file:/afs/cern.ch/user/s/scodella/public/Databases/Barrel_1125_PG_20100313_LINKFIT_NOOMRON_BOTH_PG_Err.db'),
#DBParameters = CondCore.CondDB.CondDB_cfi.CondDBSetup.DBParameters,
#toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"),       tag = cms.string("DTAlignmentRcd")),
#                  cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"),  tag = cms.string("DTAlignmentErrorExtendedRcd")),
#                  cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd")),
#                  cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))
#                  ))
#process.es_prefer_muonAlignment = cms.ESPrefer("PoolDBESSource", "muonAlignment")

### read TrackerGeometry ### Publo cff
#process.trackerAlignment = cms.ESSource("PoolDBESSource",
#connect = cms.string("sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/CRAFT09/TrackerAlignment_2009_v1_prompt/131020-infty/TrackerAlignment_2009_v1_prompt.db"),
#DBParameters = CondCore.CondDB.CondDB_cfi.CondDBSetup.DBParameters,
#toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"), tag = cms.string("Alignments")),
#                  cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"), tag = cms.string("AlignmentErrorsExtended"))
#      ))
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")


#  read data  ----------------------------------------
# process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring('file:FileToBeRead.root'))

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(

# craft08  32K
#    '/store/relval/CMSSW_3_1_4/Cosmics/ALCARECO/CRAFT0831X_V1_RelVal_StreamMuAlGlobalCosmics-v1/0006/A27618A1-73B1-DE11-B280-001D09F24FBA.root',
#  22x MC cosmics
#'rfio:/castor/cern.ch/cms/store/caf/user/pivarski/22X/trackercosmics_MuAlGlobalCosmics/ALCARECOMuAlGlobalCosmics-tkCosmics000.root',

# beam commisioning 09 18th Fed   GR09_R_35_V2  
#       '/store/data/BeamCommissioning09/Cosmics/ALCARECO/18thFebPreProd_351p1_MuAlGlobalCosmics-v1/0015/EE100A1F-481D-DF11-A98F-002618943923.root',

# Commissioning09   lCosmics-Mar3rd   GR09_R_35_V3C::All  
#    '/store/data/Commissioning09/Cosmics/ALCARECO/MuAlGlobalCosmics-Mar3rdReReco_v1/0015/AA3306FF-B62A-DF11-969A-00163E0101E1.root',

# 1st repro                         CRAFT09
#    '/store/data/CRAFT09/Cosmics/ALCARECO/StreamMuAlGlobalCosmics-CRAFT09_R_V4_CosmicsSeq_v1/0012/5A3DA2E8-ABBA-DE11-B18E-0026189437E8.root',

# Cosmic 2010 
#    'file:/tmp/spiridon/4C71BC21-5E20-DF11-8177-001D09F290CE.root',
    '/store/data/Commissioning10/Cosmics/ALCARECO/v3/000/128/762/6234035C-DC1F-DF11-BF68-001D09F25456.root',
#    '/store/data/Commissioning10/Cosmics/ALCARECO/v3/000/128/899/F6A05CFE-B620-DF11-8ACA-001D09F2545B.root',
#    '/store/data/Commissioning10/Cosmics/ALCARECO/v3/000/128/755/DC552B38-C41F-DF11-AA00-000423D6B444.root',
#    'file:FileToRead.root',

#  314 MC IsolatedMu
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0005/7812C1DF-0292-DE11-8BBF-00304867920C.root',
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/EE2DFA1B-0192-DE11-A055-001731AF6847.root',
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/EC7F4503-0192-DE11-A068-001731AF6B7D.root',
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/E8777D22-0192-DE11-B00C-001731AF68B9.root',
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/DE415319-0192-DE11-9E78-001731AF66F1.root',
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/DCD1E223-0192-DE11-B50C-001731EF61B4.root',
# 31k
    ))
