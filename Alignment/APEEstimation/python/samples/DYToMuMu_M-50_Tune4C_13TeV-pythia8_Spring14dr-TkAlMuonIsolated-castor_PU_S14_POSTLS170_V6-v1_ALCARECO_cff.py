import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/ALCARECO/TkAlZMuMu-PU20bx25_PHYS14_25_V1-v1/00000/44AE8811-716C-E411-A012-002590DB923C.root' ]);
       #'/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/00909ABE-28EA-E311-B4BE-001E67398043.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/0AB710B7-29EA-E311-B5EA-001E67396D56.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/124D79A9-5EE9-E311-89DB-002481E1511E.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/183CF5AD-52E9-E311-9C7B-0025B3E05DCA.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/1E3A7EFF-50E9-E311-B681-0025B3E0653C.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/20E35088-45E9-E311-B672-0025B31E3CC0.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/20E56621-5DE9-E311-8346-0025B3E05D0A.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/28CF302A-29EA-E311-B867-001E67396BB7.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/2A81747B-2BEA-E311-B01A-001E67396AC2.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/40E1BCAB-28EA-E311-8CFE-0025B3E063E8.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/4864517A-2BEA-E311-B8BC-001E67396C9D.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/4A942EF5-28EA-E311-B5A7-002481E14FBA.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/4E0582EA-29EA-E311-8833-001E67398A43.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/52C7B804-90EA-E311-BDF6-002590200938.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/54EE79A5-28EA-E311-AB54-0025B3E05E0A.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/5CF8B3A0-55E9-E311-B891-001E67397AD0.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/663D92E9-52E9-E311-9434-0025B3E0653C.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/668650A1-29EA-E311-AC77-002481E15110.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/66A41ADE-50E9-E311-BA1B-001E67398E12.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/6E6D42F5-2BEA-E311-9764-001E67397E13.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/7C1C6115-54E9-E311-AECC-002590200AD8.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/84E374AD-28EA-E311-97E0-002590A3716C.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/883C2659-28EA-E311-A7DB-002590A3C97E.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/A2D9372A-44E9-E311-8365-002481E14FEE.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/A6532D4F-28EA-E311-A3FE-001E67398043.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/AE0495A5-48E9-E311-9FCA-002481E75ED0.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/AE641D67-44E9-E311-BE6A-001E6739665D.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/B0904F54-28EA-E311-A9AA-001E673987D2.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/B231EAA6-52E9-E311-967C-002590A8881C.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/B629F4C6-54E9-E311-80E0-0025B3E0653C.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/BE6DF48C-28EA-E311-83C0-001E67396C9D.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/C0DC15A1-55E9-E311-A0DF-002590200A88.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/C20FEA21-5DE9-E311-83F1-0025902008C8.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/C2C51631-29EA-E311-A39A-002590A88812.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/C4495620-29EA-E311-B404-001E67397E1D.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/C85596F5-54E9-E311-84FC-002590A81DAC.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/D68F4586-45E9-E311-8D56-002590A831AA.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/DA931499-28EA-E311-B387-002481E15258.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/DE6CA403-48E9-E311-94A2-002590A37116.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/E8BED2DA-54E9-E311-A0AA-002590A887F8.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/EC1FF5FB-47E9-E311-8DDA-002590A37106.root',
      # '/store/mc/Spring14dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/ALCARECO/TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1/00000/ECC7ECAE-78E8-E311-8069-001E67398228.root' ] );


secFiles.extend( [
               ] )


