import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

heavyFlavorValidationHarvesting = DQMEDHarvester("HeavyFlavorHarvesting",
  MyDQMrootFolder = cms.untracked.string('HLT/BPH/HLT/HLT_Mu5'),
  Efficiencies = cms.untracked.VPSet(
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globMuon_genEtaPt","genMuon_genEtaPt","effGlobGen_genEtaPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathMuon_recoEtaPt","globMuon_recoEtaPt","effPathGlob_recoEtaPt") ),

    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globMuon_genEtaPhi","genMuon_genEtaPhi","effGlobGen_genEtaPhi") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathMuon_recoEtaPhi","globMuon_recoEtaPhi","effPathGlob_recoEtaPhi") ),

#################  DOUBLE  ETA  PT  ######################
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globDimuon_genEtaPt","genDimuon_genEtaPt","effGlobDigenAND_genEtaPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathDimuon_recoEtaPt","globDimuon_recoEtaPt","effPathDiglobOR_recoEtaPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("diPathDimuon_recoEtaPt","globDimuon_recoEtaPt","effPathDiglobAND_recoEtaPt") ),

#################  DOUBLE  RAPIDITY  PT  ######################
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globDimuon_genRapPt","genDimuon_genRapPt","effGlobDigenAND_genRapPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathDimuon_recoRapPt","globDimuon_recoRapPt","effPathDiglobOR_recoRapPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("diPathDimuon_recoRapPt","globDimuon_recoRapPt","effPathDiglobAND_recoRapPt") ),
    
#################  DOUBLE  PT  DR  ######################
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globDimuon_genPtDR","genDimuon_genPtDR","effGlobDigenAND_genPtDR") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathDimuon_recoPtDR","globDimuon_recoPtDR","effPathDiglobOR_recoPtDR") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("diPathDimuon_recoPtDR","globDimuon_recoPtDR","effPathDiglobAND_recoPtDR") ),

#################  DOUBLE  ETA  PT  ######################
#    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("globDimuon_genPtDRpos","genDimuon_genPtDRpos","effGlobDigenAND_genPtDRpos") ),
#    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("pathDimuon_recoPtDRpos","globDimuon_recoPtDRpos","effPathDiglobOR_recoPtDRpos") ),
#    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("diPathDimuon_recoPtDRpos","globDimuon_recoPtDRpos","effPathDiglobAND_recoPtDRpos") ),

    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("resultDimuon_recoEtaPt","globDimuon_recoEtaPt","effResultDiglob_recoEtaPt") ),
    cms.untracked.PSet( NumDenEffMEnames = cms.untracked.vstring("resultDimuon_recoRapPt","globDimuon_recoRapPt","effResultDiglob_recoRapPt") )

  )
)
