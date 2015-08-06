import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/FE5FC8EB-B4CE-E011-BE25-00261894383B.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/F4CC08F8-B4CE-E011-B3DE-003048678B84.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/EA8344E2-B4CE-E011-9307-001A92811742.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/E648A6DD-B4CE-E011-9227-003048678FF2.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/E2533EEE-B4CE-E011-A05E-003048678BEA.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/E0229DF4-B4CE-E011-87B1-0018F3D096F0.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/D80640D5-B4CE-E011-9BA1-0026189438A5.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/D26BF5E3-B4CE-E011-9CF4-001A9281173E.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/CCA441E2-B4CE-E011-9F58-003048678BE6.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/CC93C7D6-B4CE-E011-96BC-0018F3D095F0.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/CA2292F7-B4CE-E011-A8C6-003048679214.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/C6BF67D9-B4CE-E011-B3D6-0018F3D096C0.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/C60B6CDE-B4CE-E011-836B-003048678B76.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/C2B2E917-B5CE-E011-9AAC-003048678A88.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/C06866EC-B4CE-E011-B4F1-003048678B06.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/BA4C11EB-B4CE-E011-9F5B-003048678B76.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/B800ADF2-B4CE-E011-A13C-001A92971B8C.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/A8D49ADC-B4CE-E011-B3BF-0018F3D0960C.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/9C9E63E2-B4CE-E011-957F-003048679008.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/9A366FE1-B4CE-E011-918D-002618943905.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/8CAECCDD-B4CE-E011-BD1D-003048D15DB6.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/889D25EF-B4CE-E011-83CE-002618943924.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/880737EC-B4CE-E011-BE89-003048678EE2.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/84D42DE6-B4CE-E011-B8A3-001A9281170C.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/848D79E6-B4CE-E011-AE38-0026189438AE.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/7AF205F0-B4CE-E011-935C-002354EF3BDF.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/7031A3E6-B4CE-E011-B2E4-003048679164.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/6E488ED7-B4CE-E011-B520-00304867C04E.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/6CC973DC-B4CE-E011-9277-0026189438DC.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/6AEC0BD8-B4CE-E011-8BF2-001A928116B0.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/5ED838E4-B4CE-E011-9001-002618943940.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/5CC7E7F2-B4CE-E011-A5E3-00304867C0F6.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/543B7DE2-B4CE-E011-9BF4-00261894393D.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/5435E4D6-B4CE-E011-964F-003048678E80.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/4009BFE1-B4CE-E011-A2EA-003048679012.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/36298BF0-B4CE-E011-9DED-00261894386F.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/2E3C4AEF-B4CE-E011-961C-0026189437EC.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/265A1B17-B5CE-E011-9064-00304867904E.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/24D5BAE5-B4CE-E011-BBBB-003048678FF6.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/1ED0EBD8-B4CE-E011-930C-00261894397A.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/1C75DADD-B4CE-E011-AB40-00304867BFAA.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/1C4A2B12-B5CE-E011-ACA3-00304867C026.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/18B37706-B5CE-E011-98FC-0030486792B4.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/16E33EE9-B4CE-E011-81C4-003048678B18.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/161A1B17-B5CE-E011-AFF5-00304867904E.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/125E47DD-B4CE-E011-A30D-003048678B88.root',
       '/store/mc/Summer11/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/ALCARECO/TkAlMuonIsolated-PU_S4_START42_V12-v1/0000/08F19EE3-B4CE-E011-B5D6-0026189438DA.root' ] );


secFiles.extend( [
               ] )

