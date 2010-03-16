import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalAlignment")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process = cms.Process("write")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))

process.load("Alignment.CommonAlignmentProducer.GlobalTrackerMuonAlignment_cfi")
process.GlobalTrackerMuonAlignment.tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics:TrackerOnly")
process.GlobalTrackerMuonAlignment.muons = cms.InputTag("ALCARECOMuAlGlobalCosmics:StandAlone")
process.GlobalTrackerMuonAlignment.gmuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")
process.GlobalTrackerMuonAlignment.smuons = cms.InputTag("ALCARECOMuAlGlobalCosmics:SelectedMuons")
process.GlobalTrackerMuonAlignment.Propagator = cms.string("SteppingHelixPropagator")
process.GlobalTrackerMuonAlignment.cosmics = cms.bool(True)
process.GlobalTrackerMuonAlignment.writeDB = cms.untracked.bool(False)


#process.GlobalAlignmentAnalyzer = cms.EDAnalyzer("GlobalTrackerMuonAlignment",
# isolated muon
#                              tracks = cms.untracked.InputTag("ALCARECOMuAlCalIsolatedMu:TrackerOnly"),
#                              muons = cms.untracked.InputTag("ALCARECOMuAlCalIsolatedMu:StandAlone"),
#                              gmuons = cms.untracked.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon"),
#                              smuons = cms.untracked.InputTag("ALCARECOMuAlCalIsolatedMu:SelectedMuons"),
# global cosmic 
#                              tracks = cms.untracked.InputTag("ALCARECOMuAlGlobalCosmics:TrackerOnly"),
#                              muons = cms.untracked.InputTag("ALCARECOMuAlGlobalCosmics:StandAlone"),
#                              gmuons = cms.untracked.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon"),
#                              smuons = cms.untracked.InputTag("ALCARECOMuAlGlobalCosmics:SelectedMuons"),
# propagator
#                              Propagator = cms.string("SteppingHelixPropagator"))

process.p = cms.Path(process.GlobalTrackerMuonAlignment)

#process.dump=cms.EDAnalyzer('EventContentAnalyzer')
#process.p = cms.Path(process.GlobalAlignmentAnalyzer*process.dump)

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
# Geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# import of standard configurations Cosmics
process.load('Configuration/EventContent/EventContentCosmics_cff')

#                             GlobalTag
# 3_6_0
#process.GlobalTag.globaltag = 'START36_V2::All'
#process.GlobalTag.globaltag = 'DESIGN36_V2::All'
process.GlobalTag.globaltag = 'MC_36Y_V2::All'
# isolated muon
#process.GlobalTag.globaltag = "MC_31X_V5::All"
# 314 Cosmics
#process.GlobalTag.globaltag = "COSMMC_22X_V6::All"
# craft08
#process.GlobalTag.globaltag = 'CRAFT0831X_V1::All'
#process.GlobalTag.globaltag = 'STARTUP31X_V4::All'
# craft09
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
#process.GlobalTag.globaltag = 'CRAFT09_R_V6::All'
 
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
# 314 cosmics
    '/store/relval/CMSSW_3_1_4/RelValCosmics/ALCARECO/STARTUP31X_V2_StreamMuAlGlobalCosmics-v1/0006/542FDCE6-72B1-DE11-A910-001D09F2A465.root',
#  22x MC cosmics
#'rfio:/castor/cern.ch/cms/store/caf/user/pivarski/22X/trackercosmics_MuAlGlobalCosmics/ALCARECOMuAlGlobalCosmics-tkCosmics000.root',
#  314 MC IsolatedMu
#       '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0005/7812C1DF-0292-DE11-8BBF-00304867920C.root',
#       '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/EE2DFA1B-0192-DE11-A055-001731AF6847.root',
#       '/store/mc/Summer09/InclusiveMu15/ALCARECO/MC_31X_V3_StreamMuAlCalIsolatedMu-v2/0004/EC7F4503-0192-DE11-A068-001731AF6B7D.root',
    ))
