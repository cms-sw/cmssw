import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalAlignment")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process = cms.Process("write")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))

process.load("Alignment.CommonAlignmentProducer.GlobalTrackerMuonAlignment_cfi")
# cosmic muon
process.GlobalTrackerMuonAlignment.tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics:TrackerOnly")
process.GlobalTrackerMuonAlignment.muons = cms.InputTag("ALCARECOMuAlGlobalCosmics:StandAlone")
process.GlobalTrackerMuonAlignment.gmuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")
process.GlobalTrackerMuonAlignment.smuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:SelectedMuons")
# isolated muon 
#process.GlobalTrackerMuonAlignment.tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:TrackerOnly")
#process.GlobalTrackerMuonAlignment.muons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:StandAlone")
#process.GlobalTrackerMuonAlignment.gmuons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
#process.GlobalTrackerMuonAlignment.smuons = cms.InputTag("ALCARECOMuAlCalIsolatedMu:SelectedMuons")

process.GlobalTrackerMuonAlignment.Propagator = cms.string("SteppingHelixPropagator")

#process.GlobalTrackerMuonAlignment.debug = cms.untracked.bool(True)
process.GlobalTrackerMuonAlignment.cosmics = cms.bool(True)
#process.GlobalTrackerMuonAlignment.isolated = cms.bool(True)
process.GlobalTrackerMuonAlignment.writeDB = cms.untracked.bool(True)

process.p = cms.Path(process.GlobalTrackerMuonAlignment)

#process.dump=cms.EDAnalyzer('EventContentAnalyzer')
#process.p = cms.Path(process.GlobalTrackerMuonAlignment*process.dump)

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
### Force Magnetic Field B=3.8T (maybe needed for runs: 12476-127764 of cosmics 2010)
#process.load("Configuration/StandardSequences/MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# import of standard configurations Cosmics
process.load('Configuration/EventContent/EventContentCosmics_cff')

#                             GlobalTag
# 3_6_0
#process.GlobalTag.globaltag = 'START36_V2::All'
#process.GlobalTag.globaltag = 'DESIGN36_V2::All'
#process.GlobalTag.globaltag = 'GR09_R_36X_V1::All'
#process.GlobalTag.globaltag = 'MC_36Y_V2::All'
# isolated muon
#process.GlobalTag.globaltag = "MC_31X_V5::All"
# 314 Cosmics
#process.GlobalTag.globaltag = "COSMMC_22X_V6::All"
# craft08
#process.GlobalTag.globaltag = 'CRAFT0831X_V1::All'
#process.GlobalTag.globaltag = 'STARTUP31X_V4::All'
# craft09 small samples
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
#process.GlobalTag.globaltag = 'CRAFT09_R_V6::All'
#process.GlobalTag.globaltag = 'GR09_R_35_V2::ALL'
#process.GlobalTag.globaltag = 'GR09_R_35_V3C::All'

# 1st repro craft09
#process.GlobalTag.globaltag = 'CRAFT09_R_V10::All'

# cosmic 2010
#process.GlobalTag.globaltag = 'GR09_R_35X_V3::All'
process.GlobalTag.globaltag = 'GR09_R_36X_V1::All'

## propagator
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    # Writing to oracle needs the following shell variable setting (in zsh):
    # export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    # string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:output.db'),
    # untracked uint32 authenticationMethod = 1
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    ))
)

process.CondDBSetup.DBParameters.messageLevel = 2

# process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring('file:FileToBeRead.root'))

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(

# craft08  32K
#    'file:/tmp/spiridon/CRAFT08/A27618A1-73B1-DE11-B280-001D09F24FBA.root', 
#    '/store/relval/CMSSW_3_1_4/Cosmics/ALCARECO/CRAFT0831X_V1_RelVal_StreamMuAlGlobalCosmics-v1/0006/A27618A1-73B1-DE11-B280-001D09F24FBA.root',
#    '/store/FileToBeRead.root',
#  22x MC cosmics
#'rfio:/castor/cern.ch/cms/store/caf/user/pivarski/22X/trackercosmics_MuAlGlobalCosmics/ALCARECOMuAlGlobalCosmics-tkCosmics000.root',

# beam commisioning 09 18th Fed   GR09_R_35_V2  
#       '/store/data/BeamCommissioning09/Cosmics/ALCARECO/18thFebPreProd_351p1_MuAlGlobalCosmics-v1/0015/EE100A1F-481D-DF11-A98F-002618943923.root',

# Commissioning09   lCosmics-Mar3rd   GR09_R_35_V3C::All  
#    '/store/data/Commissioning09/Cosmics/ALCARECO/MuAlGlobalCosmics-Mar3rdReReco_v1/0015/AA3306FF-B62A-DF11-969A-00163E0101E1.root',

# 1st repro                         CRAFT09
#    '/store/data/CRAFT09/Cosmics/ALCARECO/StreamMuAlGlobalCosmics-CRAFT09_R_V4_CosmicsSeq_v1/0012/5A3DA2E8-ABBA-DE11-B18E-0026189437E8.root',

# Cosmic 2010  end   run info page 1-23
    '/store/data/Commissioning10/Cosmics/ALCARECO/v3/000/128/762/6234035C-DC1F-DF11-BF68-001D09F25456.root',

#  314 MC IsolatedMu
#    '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0005/7812C1DF-0292-DE11-8BBF-00304867920C.root',
    ))
