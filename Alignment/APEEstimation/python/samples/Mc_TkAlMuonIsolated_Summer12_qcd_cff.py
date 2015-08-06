import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [

'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/387445F6-65F1-E111-A76A-00248C65A3EC.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/6E97F5FB-65F1-E111-9CB0-003048D15E02.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/728B9914-66F1-E111-BB8E-0018F3D09644.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/745D8509-66F1-E111-9707-0026189438F2.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/7E956014-66F1-E111-A601-0018F3D0960A.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/8EBD6815-66F1-E111-95F4-00261894383F.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/A2DB2913-66F1-E111-8E2A-001A92810AF4.root',
'/store/mc/Summer12_DR53X/QCD_Pt_20_MuEnrichedPt_15_TuneZ2star_8TeV_pythia6/ALCARECO/TkAlMuonIsolated-PU_S10_START53_V7A-v1/00000/E60DE6F4-65F1-E111-A0C3-003048678E80.root' ] )

secFiles.extend( [
               ] )
