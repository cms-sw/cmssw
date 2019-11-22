import FWCore.ParameterSet.Config as cms

##################### Updated tau collection with MVA-based tau-Ids rerun #######
# Used only in some eras
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv1_cff import run2_nanoAOD_94XMiniAODv1

### MVAIso 2017v2
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_VVLoose"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_VLoose")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_Loose")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_Medium")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_Tight")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_VTight")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_VVTight")
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTSeq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw
    + patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT
    + patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT
)
## DBnewDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT"), # name of the training you want to use
   mvaOpt = cms.string("DBnewDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_VVLoose"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_VLoose")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_Loose")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_Medium")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_Tight")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_VTight")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_VVTight")
# MVAIso DBnewDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTSeq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw
    + patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT
    + patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT
)
## DBoldDMdR0p3
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   srcChargedIsoPtSum = cms.string('chargedIsoPtSumdR03'),
   srcFootprintCorrection = cms.string('footprintCorrectiondR03'),
   srcNeutralIsoPtSum = cms.string('neutralIsoPtSumdR03'),
   srcPUcorrPtSum = cms.string('puCorrPtSum'),
   srcPhotonPtSumOutsideSignalCone = cms.string('photonPtSumOutsideSignalConedR03'),
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_VVLoose"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_VLoose")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_Loose")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_Medium")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_Tight")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_VTight")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_VVTight")
# MVAIso DBoldDMdR0p3 Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTSeq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw
    + patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT
    + patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT
)
### MVAIso 2017v1 for Nano on top of MiniAODv1
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1 = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1 = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff95"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff90")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff80")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff70")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff60")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff50")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2017v1.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff40")
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1Seq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1
    + patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2017v1
    + patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2017v1
)
### MVAIso 2015 for Nano on top of MiniAODv2
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015 = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLT"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   verbosity = cms.int32(0)
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015 = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff90"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2015.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff80")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2015.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff70")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2015.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff60")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2015.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff50")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2015.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff40")
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015Seq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015
    + patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015
    + patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2015
    + patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2015
    + patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2015
    + patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2015
    + patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2015
)


### Define new anit-e discriminants (2018)
antiElectronDiscrMVA6_version = "MVA"
## Raw
from RecoTauTag.RecoTau.PATTauDiscriminationAgainstElectronMVA6_cfi import patTauDiscriminationAgainstElectronMVA6
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
patTauDiscriminationByElectronRejectionMVA62018Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    Prediscriminants = noPrediscriminants, #already selected for MiniAOD
    vetoEcalCracks = False, #keep tau candidates in EB-EE cracks
    mvaName_NoEleMatch_wGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_BL',
    mvaName_NoEleMatch_wGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_EC',
    mvaName_NoEleMatch_woGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_BL',
    mvaName_NoEleMatch_woGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_EC',
    mvaName_wGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_BL',
    mvaName_wGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_EC',
    mvaName_woGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_BL',
    mvaName_woGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_EC'
)
## anti-e 2018 WPs
from RecoTauTag.RecoTau.PATTauDiscriminantCutMultiplexer_cfi import patTauDiscriminantCutMultiplexer
# VLoose
patTauDiscriminationByVLooseElectronRejectionMVA62018 = patTauDiscriminantCutMultiplexer.clone(
    PATTauProducer = patTauDiscriminationByElectronRejectionMVA62018Raw.PATTauProducer,
    Prediscriminants = patTauDiscriminationByElectronRejectionMVA62018Raw.Prediscriminants,
    toMultiplex = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw"),
    key = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw","category"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_BL_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(2),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_BL_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(5),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_BL_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(7),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_BL_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(8),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_EC_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(10),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_EC_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(13),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_EC_VLoose'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(15),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_EC_VLoose'),
            variable = cms.string('pt')
        )
    )
)
# Loose
patTauDiscriminationByLooseElectronRejectionMVA62018 = patTauDiscriminationByVLooseElectronRejectionMVA62018.clone()
for m in patTauDiscriminationByLooseElectronRejectionMVA62018.mapping:
    m.cut = m.cut.value().replace("VLoose", "Loose")
# Medium
patTauDiscriminationByMediumElectronRejectionMVA62018 = patTauDiscriminationByVLooseElectronRejectionMVA62018.clone()
for m in patTauDiscriminationByMediumElectronRejectionMVA62018.mapping:
    m.cut = m.cut.value().replace("VLoose", "Medium")
# Tight
patTauDiscriminationByTightElectronRejectionMVA62018 = patTauDiscriminationByVLooseElectronRejectionMVA62018.clone()
for m in patTauDiscriminationByTightElectronRejectionMVA62018.mapping:
    m.cut = m.cut.value().replace("VLoose", "Tight")
# VTight
patTauDiscriminationByVTightElectronRejectionMVA62018 = patTauDiscriminationByVLooseElectronRejectionMVA62018.clone()
for m in patTauDiscriminationByVTightElectronRejectionMVA62018.mapping:
    m.cut = m.cut.value().replace("VLoose", "VTight")
### Define v1 anit-e discriminants (2015)
antiElectronDiscrMVA6v1_version = "MVA6v1"
## Raw
patTauDiscriminationByElectronRejectionMVA62015Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    Prediscriminants = noPrediscriminants, #already selected for MiniAOD
    vetoEcalCracks = True, #don't keep tau candidates in EB-EE cracks for v1
    mvaName_NoEleMatch_wGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_NoEleMatch_wGwoGSF_BL',
    mvaName_NoEleMatch_wGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_NoEleMatch_wGwoGSF_EC',
    mvaName_NoEleMatch_woGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_NoEleMatch_woGwoGSF_BL',
    mvaName_NoEleMatch_woGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_NoEleMatch_woGwoGSF_EC',
    mvaName_wGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_wGwGSF_BL',
    mvaName_wGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_wGwGSF_EC',
    mvaName_woGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_woGwGSF_BL',
    mvaName_woGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6v1_version+'_gbr_woGwGSF_EC'
)
## anti-e v1 WPs
# VLoose
patTauDiscriminationByVLooseElectronRejectionMVA62015 = patTauDiscriminationByVLooseElectronRejectionMVA62018.clone(
    PATTauProducer = patTauDiscriminationByElectronRejectionMVA62015Raw.PATTauProducer,
    Prediscriminants = patTauDiscriminationByElectronRejectionMVA62015Raw.Prediscriminants,
    toMultiplex = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62015Raw"),
    key = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62015Raw","category")
)
for m in patTauDiscriminationByVLooseElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace(antiElectronDiscrMVA6_version, antiElectronDiscrMVA6v1_version + "_gbr").replace("VLoose", "WPEff99")
# Loose
patTauDiscriminationByLooseElectronRejectionMVA62015 = patTauDiscriminationByVLooseElectronRejectionMVA62015.clone()
for m in patTauDiscriminationByLooseElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace("WPEff99", "WPEff96")
# Medium
patTauDiscriminationByMediumElectronRejectionMVA62015 = patTauDiscriminationByVLooseElectronRejectionMVA62015.clone()
for m in patTauDiscriminationByMediumElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace("WPEff99", "WPEff91")
# Tight
patTauDiscriminationByTightElectronRejectionMVA62015 = patTauDiscriminationByVLooseElectronRejectionMVA62015.clone()
for m in patTauDiscriminationByTightElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace("WPEff99", "WPEff85")
# VTight
patTauDiscriminationByVTightElectronRejectionMVA62015 = patTauDiscriminationByVLooseElectronRejectionMVA62015.clone()
for m in patTauDiscriminationByVTightElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace("WPEff99", "WPEff79")
### Put all anti-e tau-IDs into a sequence
_patTauDiscriminationByElectronRejection2018Seq = cms.Sequence(
    patTauDiscriminationByElectronRejectionMVA62018Raw
    +patTauDiscriminationByVLooseElectronRejectionMVA62018
    +patTauDiscriminationByLooseElectronRejectionMVA62018
    +patTauDiscriminationByMediumElectronRejectionMVA62018
    +patTauDiscriminationByTightElectronRejectionMVA62018
    +patTauDiscriminationByVTightElectronRejectionMVA62018
)
_patTauDiscriminationByElectronRejection2015Seq = cms.Sequence(
    patTauDiscriminationByElectronRejectionMVA62015Raw
    +patTauDiscriminationByVLooseElectronRejectionMVA62015
    +patTauDiscriminationByLooseElectronRejectionMVA62015
    +patTauDiscriminationByMediumElectronRejectionMVA62015
    +patTauDiscriminationByTightElectronRejectionMVA62015
    +patTauDiscriminationByVTightElectronRejectionMVA62015
)
patTauDiscriminationByElectronRejectionSeq = _patTauDiscriminationByElectronRejection2015Seq.copy()
(~run2_miniAOD_80XLegacy).toReplaceWith(patTauDiscriminationByElectronRejectionSeq,
                      _patTauDiscriminationByElectronRejection2018Seq)


### put all new MVA tau-Id stuff to one Sequence
_patTauMVAIDsSeq2017v2 = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTSeq
    +patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTSeq
    +patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTSeq
    +patTauDiscriminationByElectronRejectionSeq
)
patTauMVAIDsSeq = _patTauMVAIDsSeq2017v2.copy()
patTauMVAIDsSeq += patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015Seq

_patTauMVAIDsSeqWith2017v1 = _patTauMVAIDsSeq2017v2.copy()
_patTauMVAIDsSeqWith2017v1 += patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1Seq
for era in [run2_nanoAOD_94XMiniAODv1,]:
    era.toReplaceWith(patTauMVAIDsSeq,_patTauMVAIDsSeqWith2017v1)

# embed new MVA tau-Ids into new tau collection
slimmedTausUpdated = cms.EDProducer("PATTauIDEmbedder",
    src = cms.InputTag('slimmedTaus'),
    tauIDSources = cms.PSet() # PSet defined below in era dependent way
)
_tauIDSources2017v2 = cms.PSet(
        #oldDM
        byIsolationMVArun2v1DBoldDMwLTraw2017v2 = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
        byVVLooseIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT'),
        byVLooseIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT'),
        byLooseIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT'),
        byMediumIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT'),
        byTightIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT'),
        byVTightIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT'),
        byVVTightIsolationMVArun2v1DBoldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT'),
        #newDM
        byIsolationMVArun2v1DBnewDMwLTraw2017v2 = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
        byVVLooseIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT'),
        byVLooseIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT'),
        byLooseIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT'),
        byMediumIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT'),
        byTightIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT'),
        byVTightIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT'),
        byVVTightIsolationMVArun2v1DBnewDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT'),
        #oldDMdR0p3
        byIsolationMVArun2v1DBdR03oldDMwLTraw2017v2 = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw'),
        byVVLooseIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byVLooseIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byLooseIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byMediumIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byTightIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byVTightIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT'),
        byVVTightIsolationMVArun2v1DBdR03oldDMwLT2017v2 = cms.InputTag('patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT'),
)
_tauIDSources2017v1 = cms.PSet(
    byIsolationMVArun2v1DBoldDMwLTraw2017v1 = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1'),
    byVVLooseIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT2017v1'),
    byVLooseIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2017v1'),
    byLooseIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2017v1'),
    byMediumIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2017v1'),
    byTightIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2017v1'),
    byVTightIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2017v1'),
    byVVTightIsolationMVArun2v1DBoldDMwLT2017v1 = cms.InputTag('patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2017v1')
)
_tauIDSourcesWith2017v1 = cms.PSet(
    _tauIDSources2017v2.clone(),
    _tauIDSources2017v1
)
_tauIDSources2015 = cms.PSet(
    byIsolationMVArun2v1DBoldDMwLTraw2015 = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015'),
    byVLooseIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT2015'),
    byLooseIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT2015'),
    byMediumIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT2015'),
    byTightIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT2015'),
    byVTightIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT2015'),
    byVVTightIsolationMVArun2v1DBoldDMwLT2015 = cms.InputTag('patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT2015')
)
_tauIDSourcesWith2015 = cms.PSet(
    _tauIDSources2017v2.clone(),
    _tauIDSources2015
)
slimmedTausUpdated.tauIDSources=_tauIDSourcesWith2015

for era in [run2_nanoAOD_94XMiniAODv1,]:
    era.toModify(slimmedTausUpdated,
                 tauIDSources = _tauIDSourcesWith2017v1
    )

_antiETauIDSources2018 = cms.PSet(
    againstElectronMVA6Raw2018 = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw"),
    againstElectronMVA6category2018 = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw","category"),
    againstElectronVLooseMVA62018 = cms.InputTag("patTauDiscriminationByVLooseElectronRejectionMVA62018"),
    againstElectronLooseMVA62018 = cms.InputTag("patTauDiscriminationByLooseElectronRejectionMVA62018"),
    againstElectronMediumMVA62018 = cms.InputTag("patTauDiscriminationByMediumElectronRejectionMVA62018"),
    againstElectronTightMVA62018 = cms.InputTag("patTauDiscriminationByTightElectronRejectionMVA62018"),
    againstElectronVTightMVA62018 = cms.InputTag("patTauDiscriminationByVTightElectronRejectionMVA62018"),
)
_tauIDSourcesWithAntiE2018 = cms.PSet(
    slimmedTausUpdated.tauIDSources.clone(),
    _antiETauIDSources2018
)
_antiETauIDSources2015 = cms.PSet(
    againstElectronMVA6Raw2015 = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62015Raw"),
    againstElectronMVA6category2015 = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62015Raw","category"),
    againstElectronVLooseMVA62015 = cms.InputTag("patTauDiscriminationByVLooseElectronRejectionMVA62015"),
    againstElectronLooseMVA62015 = cms.InputTag("patTauDiscriminationByLooseElectronRejectionMVA62015"),
    againstElectronMediumMVA62015 = cms.InputTag("patTauDiscriminationByMediumElectronRejectionMVA62015"),
    againstElectronTightMVA62015 = cms.InputTag("patTauDiscriminationByTightElectronRejectionMVA62015"),
    againstElectronVTightMVA62015 = cms.InputTag("patTauDiscriminationByVTightElectronRejectionMVA62015")
)
_tauIDSourcesWithAntiE2015 = cms.PSet(
    slimmedTausUpdated.tauIDSources.clone(),
    _antiETauIDSources2015
)
slimmedTausUpdated.tauIDSources=_tauIDSourcesWithAntiE2015
(~run2_miniAOD_80XLegacy).toModify(slimmedTausUpdated,
                 tauIDSources = _tauIDSourcesWithAntiE2018
    )


patTauMVAIDsSeq += slimmedTausUpdated

