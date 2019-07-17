import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TTrackMatch.L1TkObjectAnalyzer_cfi import l1TrkObjAnaysis
MuonEff                 = l1TrkObjAnaysis.clone()
MuonEff.ObjectType      = cms.string("Muon")
MuonEff.AnalysisOption  = cms.string("Efficiency")
 
MuonRate                = l1TrkObjAnaysis.clone()
MuonRate.ObjectType     = cms.string("Muon")
MuonRate.AnalysisOption = cms.string("Rate")

PhotonEff                 = l1TrkObjAnaysis.clone()
PhotonEff.ObjectType      = cms.string("Photon")
PhotonEff.AnalysisOption  = cms.string("Efficiency")

PhotonRate                = l1TrkObjAnaysis.clone()
PhotonRate.ObjectType     = cms.string("Photon")
PhotonRate.AnalysisOption = cms.string("Rate")

ElectronEff                 = l1TrkObjAnaysis.clone()
ElectronEff.ObjectType      = cms.string("Electron")
ElectronEff.AnalysisOption  = cms.string("Efficiency")

ElectronRate                = l1TrkObjAnaysis.clone()
ElectronRate.ObjectType     = cms.string("Electron")
ElectronRate.AnalysisOption = cms.string("Rate")

IsoElectronEff                      = l1TrkObjAnaysis.clone()
IsoElectronEff.L1TkElectronInputTag = cms.InputTag("L1TkIsoElectrons","EG")
IsoElectronEff.ObjectType           = cms.string("Electron")
IsoElectronEff.AnalysisOption       = cms.string("Efficiency")

IsoElectronRate                      = l1TrkObjAnaysis.clone()
IsoElectronRate.L1TkElectronInputTag = cms.InputTag("L1TkIsoElectrons","EG")
IsoElectronRate.ObjectType           = cms.string("Electron")
IsoElectronRate.AnalysisOption       = cms.string("Rate")
