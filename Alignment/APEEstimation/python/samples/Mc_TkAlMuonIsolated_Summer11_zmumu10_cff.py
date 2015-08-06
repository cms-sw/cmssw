import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/mc/Summer11/DYToMuMu_M-10To20_CT10_TuneZ2_7TeV-powheg-pythia/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V11-v1/0000/DEB48244-F4CE-E011-B1D7-0024E8768D5B.root',
       '/store/mc/Summer11/DYToMuMu_M-10To20_CT10_TuneZ2_7TeV-powheg-pythia/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V11-v1/0000/66E6FE22-F4CE-E011-94FF-0026B94E284B.root',
       '/store/mc/Summer11/DYToMuMu_M-10To20_CT10_TuneZ2_7TeV-powheg-pythia/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V11-v1/0000/1EB2CD11-F4CE-E011-B7E7-0026B94E280A.root' ] );


secFiles.extend( [
               ] )

