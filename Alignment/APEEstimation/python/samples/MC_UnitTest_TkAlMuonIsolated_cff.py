import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [
"/store/mc/Run3Winter22DRPremix/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-TRKALCADesign_design_geometry_122X_mcRun3_2021_design_v9-v2/2520000/0027e6ed-2626-4ede-97a5-f0a44164b81b.root",
"/store/mc/Run3Winter22DRPremix/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-TRKALCADesign_design_geometry_122X_mcRun3_2021_design_v9-v2/2520000/00899b9d-32ab-46f2-b77b-0b0a8d666027.root",
"/store/mc/Run3Winter22DRPremix/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-TRKALCADesign_design_geometry_122X_mcRun3_2021_design_v9-v2/2520000/07851676-0c65-4630-bbab-7406defeb670.root",
"/store/mc/Run3Winter22DRPremix/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-TRKALCADesign_design_geometry_122X_mcRun3_2021_design_v9-v2/2520000/0a54c512-0f69-44e1-90bc-16da035cbe02.root",
   ] );



secFiles.extend( [
               ] )

