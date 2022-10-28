import FWCore.ParameterSet.Config as cms

##################### Updated tau collection with MVA-based tau-Ids rerun #######
# Used only in some eras
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

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
# WPs
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT = patDiscriminationByIsolationMVArun2v1.clone(
   PATTauProducer = cms.InputTag('slimmedTaus'),
   Prediscriminants = noPrediscriminants,
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
   loadMVAfromDB = cms.bool(True),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   ),
   workingPoints = cms.vstring(
      "_VVLoose",
      "_VLoose",
      "_Loose",
      "_Medium",
      "_Tight",
      "_VTight",
      "_VVTight"
   )
)
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT
)
## DBnewDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT"), # name of the training you want to use
   mvaOpt = cms.string("DBnewDMwLTwGJ") # option you want to use for your training (i.e., which variables are used to compute the BDT score)
)
# WPs
patTauDiscriminationByIsolationMVArun2v1DBnewDMwLT = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# MVAIso DBnewDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw,
    patTauDiscriminationByIsolationMVArun2v1DBnewDMwLT
)
## DBoldDMdR0p3
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ"), # option you want to use for your training (i.e., which variables are used to compute the BDT score)
   srcChargedIsoPtSum = cms.string('chargedIsoPtSumdR03'),
   srcFootprintCorrection = cms.string('footprintCorrectiondR03'),
   srcNeutralIsoPtSum = cms.string('neutralIsoPtSumdR03'),
   srcPUcorrPtSum = cms.string('puCorrPtSum'),
   srcPhotonPtSumOutsideSignalCone = cms.string('photonPtSumOutsideSignalConedR03')
)
# WPs
patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLT = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw'),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   )
)
# MVAIso DBoldDMdR0p3 Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTTask = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTraw,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLT
)
### MVAIso 2017v1 for Nano on top of MiniAODv1
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1 = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
   mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLTwGJ") # option you want to use for your training (i.e., which variables are used to compute the BDT score)
)
# WPs
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1 = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1'),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   ),
   workingPoints = cms.vstring(
      "_WPEff95",
      "_WPEff90",
      "_WPEff80",
      "_WPEff70",
      "_WPEff60",
      "_WPEff50",
      "_WPEff40"
   )
)
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1Task = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2017v1,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2017v1
)
### MVAIso 2015 for Nano on top of MiniAODv2
## DBoldDM
# Raw
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015 = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
   mvaName = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"), # name of the training you want to use
   mvaOpt = cms.string("DBoldDMwLT") # option you want to use for your training (i.e., which variables are used to compute the BDT score)
)
# WPs
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015 = patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
   toMultiplex = cms.InputTag('patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015'),
   mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization"), # normalization fo the training you want to use
   mapping = cms.VPSet(
      cms.PSet(
         category = cms.uint32(0),
         cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"), # this is the name of the working point you want to use
         variable = cms.string("pt"),
      )
   ),
   workingPoints = cms.vstring(
      "_WPEff90",
      "_WPEff80",
      "_WPEff70",
      "_WPEff60",
      "_WPEff50",
      "_WPEff40"
   )
)
# MVAIso DBoldDM Seqeunce
patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015Task = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw2015,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015
)


### Define new anit-e discriminants (2018)
antiElectronDiscrMVA6_version = "MVA"
## Raw
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import patTauDiscriminationAgainstElectronMVA6
patTauDiscriminationByElectronRejectionMVA62018Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    PATTauProducer = 'slimmedTaus',
    Prediscriminants = noPrediscriminants, #already selected for MiniAOD
    srcElectrons = 'slimmedElectrons',
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
patTauDiscriminationByElectronRejectionMVA62018 = patTauDiscriminantCutMultiplexer.clone(
    PATTauProducer = patTauDiscriminationByElectronRejectionMVA62018Raw.PATTauProducer,
    Prediscriminants = patTauDiscriminationByElectronRejectionMVA62018Raw.Prediscriminants,
    toMultiplex = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_BL'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(2),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_BL'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(5),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_BL'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(7),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_BL'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(8),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_woGwoGSF_EC'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(10),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_NoEleMatch_wGwoGSF_EC'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(13),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_woGwGSF_EC'),
            variable = cms.string('pt')
        ),
        cms.PSet(
            category = cms.uint32(15),
            cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_wGwGSF_EC'),
            variable = cms.string('pt')
        )
    ),
    rawValues = cms.vstring(
        "discriminator",
        "category"
    ),
    workingPoints = cms.vstring(
      "_VLoose",
      "_Loose",
      "_Medium",
      "_Tight",
      "_VTight"
    )
)
### Define v1 anit-e discriminants (2015)
antiElectronDiscrMVA6v1_version = "MVA6v1"
## Raw
patTauDiscriminationByElectronRejectionMVA62015Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    PATTauProducer = 'slimmedTaus',
    Prediscriminants = noPrediscriminants, #already selected for MiniAOD
    srcElectrons = 'slimmedElectrons',
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
patTauDiscriminationByElectronRejectionMVA62015 = patTauDiscriminationByElectronRejectionMVA62018.clone(
    PATTauProducer = patTauDiscriminationByElectronRejectionMVA62015Raw.PATTauProducer,
    Prediscriminants = patTauDiscriminationByElectronRejectionMVA62015Raw.Prediscriminants,
    toMultiplex = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62015Raw"),
    rawValues = cms.vstring(
        "discriminator",
        "category"
    ),
    workingPoints = cms.vstring(
      "_WPEff99",
      "_WPEff96",
      "_WPEff91",
      "_WPEff85",
      "_WPEff79"
    )
)
for m in patTauDiscriminationByElectronRejectionMVA62015.mapping:
    m.cut = m.cut.value().replace(antiElectronDiscrMVA6_version, antiElectronDiscrMVA6v1_version + "_gbr")

patTauDiscriminationByElectronRejectionTask = cms.Task(
    patTauDiscriminationByElectronRejectionMVA62015Raw,
    patTauDiscriminationByElectronRejectionMVA62015
)


### put all new MVA tau-Id stuff to one Sequence
patTauMVAIDsTask = cms.Task(
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask,
    patTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMdR0p3wLTTask,
    patTauDiscriminationByElectronRejectionTask,
    patTauDiscriminationByIsolationMVArun2v1DBoldDMwLT2015Task
)

slimmedTausUpdated = cms.EDProducer("PATTauIDEmbedder",
    src = cms.InputTag('slimmedTaus'),
    tauIDSources = cms.PSet() # PSet defined below in era dependent way
)

patTauMVAIDsTask.add(slimmedTausUpdated)
