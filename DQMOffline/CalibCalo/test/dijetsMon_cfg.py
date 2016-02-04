import FWCore.ParameterSet.Config as cms

process = cms.Process("dijetsval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre3/RelValQCD_Pt_80_120/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalDijets_v1/0001/D43D0349-800A-DE11-8751-000423D6006E.root'
))

process.load("DQMOffline.CalibCalo.MonitorHcalDiJetsAlCaReco_cfi")

process.load("DQMServices.Core.DQMStore_cfg")


process.p = cms.Path(process.MonitorHcalDiJetsAlCaReco)


