import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackSkim")


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V2P::All"

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
                            debugVerbosity = cms.untracked.uint32(0),
                            debugFlag = cms.untracked.bool(False),
                            fileNames = cms.untracked.vstring(
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_1.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_2.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_3.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_4.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_5.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_6.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_7.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_8.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_9.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_10.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_11.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_12.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_13.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_14.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_15.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_16.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_17.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_18.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_19.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_20.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_21.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_22.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_23.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_24.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_25.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_26.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_27.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_28.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_29.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_30.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_31.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_32.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_33.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_34.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_35.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_36.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_37.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_38.root',
#    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_39.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_40.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_41.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_42.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_43.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_44.root',
    'file:rfio:/castor/cern.ch/user/b/bellan/Run124020/MuonTrackSkim_45.root'
    
     ),
                            
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/MuonAnalysis/Skims/test/TrackSkim_cfg.py,v $'),
    annotation = cms.untracked.string('track skim')
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


########################## Muon tracks Filter ############################
process.trackFilter = cms.EDFilter("TrackFilter",
                                   trackLabel = cms.InputTag("generalTracks"),
                                   atLeastNTracks = cms.uint32(3) 
    )

process.tracksSkim = cms.Path(process.trackFilter)
###########################################################################



process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TrackSkim_124020.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('Track_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("tracksSkim")
    )
)

process.e = cms.EndPath(process.out)

