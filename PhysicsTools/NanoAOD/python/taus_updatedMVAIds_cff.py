import FWCore.ParameterSet.Config as cms

##################### Updated tau collection with MVA-based tau-Ids rerun #######
# Used only in some eras
from Configuration.StandardSequences.Eras import eras
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *

### MVAIso 2017v2
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw = patDiscriminationByIsolationMVArun2v1raw.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   loadMVAfromDB = cms.bool(True),
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   requireDecayMode = cms.bool(True),
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff95"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff90")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff80")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff70")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff60")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff50")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff40")
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
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"), # name of the training you want to use
   mvaOpt = cms.string("DBnewDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   requireDecayMode = cms.bool(True),
   verbosity = cms.int32(0)
)
# VVLoose WP
patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT = patDiscriminationByIsolationMVArun2v1VLoose.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
   key = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw','category'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff95"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff90")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff80")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff70")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff60")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff50")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBnewDMwLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff40")
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
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   requireDecayMode = cms.bool(True),
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
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff95"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# VLoose WP
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff90")
# Loose WP
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByLooseIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff80")
# Medium WP
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByMediumIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff70")
# Tight WP
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff60")
# VTight WP
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff50")
# VVTights WP
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMdR0p3wLT.clone()
patTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMdR0p3wLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff40")
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
   requireDecayMode = cms.bool(True),
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
   requireDecayMode = cms.bool(True),
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


### FIXME: add other tau-Ids when ready

### put all new MVA tau-Id stuff to one Sequence
patTauMVAIDsSeq = cms.Sequence(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTSeq
    +patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTSeq
    +patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTSeq
)
_patTauMVAIDsSeqWith2017v1 = patTauMVAIDsSeq.copy()
_patTauMVAIDsSeqWith2017v1 += patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1Seq
for era in [eras.run2_nanoAOD_94XMiniAODv1,eras.run2_nanoAOD_92X]:
    era.toReplaceWith(patTauMVAIDsSeq,_patTauMVAIDsSeqWith2017v1)
_patTauMVAIDsSeqWith2015 = patTauMVAIDsSeq.copy()
_patTauMVAIDsSeqWith2015 += patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015Seq
for era in [eras.run2_nanoAOD_94XMiniAODv2, eras.run2_nanoAOD_94X2016]:
    era.toReplaceWith(patTauMVAIDsSeq, _patTauMVAIDsSeqWith2015)

# embed new MVA tau-Ids into new tau collection
slimmedTausUpdated = cms.EDProducer("PATTauIDEmbedder",
    src = cms.InputTag('slimmedTaus'),
    tauIDSources = cms.PSet(
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
    slimmedTausUpdated.tauIDSources.clone(),
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
    slimmedTausUpdated.tauIDSources.clone(),
    _tauIDSources2015
)

for era in [eras.run2_nanoAOD_94XMiniAODv1,eras.run2_nanoAOD_92X]:
    era.toModify(slimmedTausUpdated,
                 tauIDSources = _tauIDSourcesWith2017v1
    )

for era in [eras.run2_nanoAOD_94X2016, eras.run2_nanoAOD_94XMiniAODv2]:
    era.toModify(slimmedTausUpdated,
                 tauIDSources=_tauIDSourcesWith2015
                )



patTauMVAIDsSeq += slimmedTausUpdated

