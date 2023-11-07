import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.Skims_DPG_cff import skimContent

from Configuration.EventContent.EventContent_cff import RECOEventContent
skimRecoContent = RECOEventContent.clone()
skimRecoContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRecoContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.EventContent.EventContent_cff import RAWEventContent
skimRawContent = RAWEventContent.clone()
skimRawContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRawContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.EventContent.EventContent_cff import RAWAODEventContent
skimRawAODContent = RAWAODEventContent.clone()
skimRawAODContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRawAODContent.outputCommands.append("drop *_*_*_SKIM")

#####################
# event splitting special skims

# select events 1, 5, 9, ...
evtSplit_Prescaler_P1 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(4),
    prescaleOffset = cms.int32(1)
)
# select events 2, 6, 10, ...
evtSplit_Prescaler_P2 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(4),
    prescaleOffset = cms.int32(2)
)
# select events 3, 7, 11, ...
evtSplit_Prescaler_P3 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(4),
    prescaleOffset = cms.int32(3)
)
# select events 4, 8, 12, ...
evtSplit_Prescaler_P4 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(4),
    prescaleOffset = cms.int32(0)
)

evtSplit_SkimPath_P1 = cms.Path(evtSplit_Prescaler_P1)
evtSplit_SkimPath_P2 = cms.Path(evtSplit_Prescaler_P2)
evtSplit_SkimPath_P3 = cms.Path(evtSplit_Prescaler_P3)
evtSplit_SkimPath_P4 = cms.Path(evtSplit_Prescaler_P4)

SKIMStreamevtSplitSkimP1 = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'evtSplitSkimP1',
    paths = (evtSplit_SkimPath_P1),
    content = skimRawContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW')
    )
SKIMStreamevtSplitSkimP2 = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'evtSplitSkimP2',
    paths = (evtSplit_SkimPath_P2),
    content = skimRawContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW')
    )
SKIMStreamevtSplitSkimP3 = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'evtSplitSkimP3',
    paths = (evtSplit_SkimPath_P3),
    content = skimRawContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW')
    )
SKIMStreamevtSplitSkimP4 = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'evtSplitSkimP4',
    paths = (evtSplit_SkimPath_P4),
    content = skimRawContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW')
    )

#####################
# Minbias generator skims
# 1) 1 muon with pt > 7, |eta|<2.5 --> 1Mu
# 2) 2 muons with pt > 2 and 2, |eta|<2.5 --> 2Mu
# 3) 2 muons with pt > 4 and 3, |eta|<2.5, opposite sign, mass [0.2,8.5] --> OS2Mu
# 4) 2 muons with pt > 4 and 4, |eta|<2.5, opposite sign, mass [4.9,5.9] --> OS2MuB
# 5) 3 muons with pt > 5/2/2, |eta|<2.5 --> 3Mu

from Configuration.EventContent.EventContent_cff import RAWSIMEventContent
MinBiasGenSkimFilter1 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(7.0),
          NumRequired = cms.int32(1),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter2 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(2.0),
          NumRequired = cms.int32(2),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter3 = cms.EDFilter("MCParticlePairFilter",
          Status = cms.untracked.vint32(1, 1),
          MinPt = cms.untracked.vdouble(4., 3.),
          MaxEta = cms.untracked.vdouble(2.5, 2.5),
          MinEta = cms.untracked.vdouble(-2.5, -2.5),
          ParticleCharge = cms.untracked.int32(-1),
          MaxInvMass = cms.untracked.double(8.5),
          MinInvMass = cms.untracked.double(0.2),
          ParticleID1 = cms.untracked.vint32(13),
          ParticleID2 = cms.untracked.vint32(13)
      )
MinBiasGenSkimFilter4 = cms.EDFilter("MCParticlePairFilter",
          Status = cms.untracked.vint32(1, 1),
          MinPt = cms.untracked.vdouble(4., 4.),
          MaxEta = cms.untracked.vdouble(2.5, 2.5),
          MinEta = cms.untracked.vdouble(-2.5, -2.5),
          ParticleCharge = cms.untracked.int32(-1),
          MaxInvMass = cms.untracked.double(5.9),
          MinInvMass = cms.untracked.double(4.9),
          ParticleID1 = cms.untracked.vint32(13),
          ParticleID2 = cms.untracked.vint32(13)
      )
MinBiasGenSkimFilter5 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(2.0),
          NumRequired = cms.int32(3),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter6 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(5.0),
          NumRequired = cms.int32(1),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )

MinBiasGenSkimSeq1Mu = cms.Sequence( MinBiasGenSkimFilter1 )
MinBiasGenSkimSeq2Mu = cms.Sequence( MinBiasGenSkimFilter2 )
MinBiasGenSkimSeqOS2Mu = cms.Sequence( MinBiasGenSkimFilter3 )
MinBiasGenSkimSeqOS2MuB = cms.Sequence( MinBiasGenSkimFilter4 )
MinBiasGenSkimSeq3Mu = cms.Sequence( MinBiasGenSkimFilter5 * MinBiasGenSkimFilter6)

MinBiasGenSkimPath1Mu = cms.Path(MinBiasGenSkimSeq1Mu )
MinBiasGenSkimPath2Mu = cms.Path( MinBiasGenSkimSeq2Mu )
MinBiasGenSkimPathOS2Mu = cms.Path( MinBiasGenSkimSeqOS2Mu )
MinBiasGenSkimPathOS2MuB = cms.Path( MinBiasGenSkimSeqOS2MuB )
MinBiasGenSkimPath3Mu = cms.Path( MinBiasGenSkimSeq3Mu )

SKIMStreamMinBiasGenSkim1Mu = cms.FilteredStream(
    responsible = 'GEN',
    name = 'MinBiasGenSkim1Mu',
    paths = ( MinBiasGenSkimPath1Mu),
    content = RAWSIMEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('GEN')
)
SKIMStreamMinBiasGenSkim2Mu = cms.FilteredStream(
    responsible = 'GEN',
    name = 'MinBiasGenSkim2Mu',
    paths = ( MinBiasGenSkimPath2Mu ),
    content = RAWSIMEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('GEN')
)
SKIMStreamMinBiasGenSkimOS2Mu = cms.FilteredStream(
    responsible = 'GEN',
    name = 'MinBiasGenSkimOS2Mu',
    paths = ( MinBiasGenSkimPathOS2Mu ),
    content = RAWSIMEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('GEN')
)
SKIMStreamMinBiasGenSkimOS2MuB = cms.FilteredStream(
    responsible = 'GEN',
    name = 'MinBiasGenSkimOS2MuB',
    paths = ( MinBiasGenSkimPathOS2MuB ),
    content = RAWSIMEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('GEN')
)
SKIMStreamMinBiasGenSkim3Mu = cms.FilteredStream(
    responsible = 'GEN',
    name = 'MinBiasGenSkim3Mu',
    paths = ( MinBiasGenSkimPath3Mu ),
    content = RAWSIMEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('GEN')
)

#####################

from Configuration.Skimming.PDWG_BPHSkim_cff import *
BPHSkimPath = cms.Path(BPHSkimSequence)
SKIMStreamBPHSkim = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'BPHSkim',
    paths = (BPHSkimPath),
    content = BPHSkim_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################

from Configuration.Skimming.PDWG_EXONoBPTXSkim_cff import *
EXONoBPTXSkimPath = cms.Path()
SKIMStreamEXONoBPTXSkim = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXONoBPTXSkim',
    paths = (EXONoBPTXSkimPath),
    content = EXONoBPTXSkim_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################

from Configuration.Skimming.PDWG_DiJetAODSkim_cff import *
diJetAveSkimPath = cms.Path(DiJetAveSkim_Trigger)
SKIMStreamDiJet = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiJet',
    paths = (diJetAveSkimPath),
    content = DiJetAveSkim_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################

from Configuration.Skimming.PDWG_TauSkim_cff import *
tauSkimBy1Path = cms.Path( tauSkim1Sequence )
tauSkimBy2Path = cms.Path( tauSkim2Sequence )
SKIMStreamTau = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Tau',
    paths = (tauSkimBy1Path),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )
SKIMStreamDiTau = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Tau',
    paths = (tauSkimBy2Path),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


#####################

from Configuration.Skimming.PDWG_HSCP_SD_cff import *
HSCPSDPath = cms.Path(HSCPSD)
SKIMStreamHSCPSD = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'HSCPSD',
    paths = (HSCPSDPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

#####################

from Configuration.Skimming.PDWG_DiPhoton_SD_cff import *
CaloIdIsoPhotonPairsPath = cms.Path(CaloIdIsoPhotonPairsFilter)
R9IdPhotonPairsPath = cms.Path(R9IdPhotonPairsFilter)
MixedCaloR9IdPhotonPairsPath = cms.Path(MixedCaloR9IdPhotonPairsFilter)
MixedR9CaloIdPhotonPairsPath = cms.Path(MixedR9CaloIdPhotonPairsFilter)

SKIMStreamDiPhoton = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiPhoton',
    paths = (CaloIdIsoPhotonPairsPath,R9IdPhotonPairsPath,MixedCaloR9IdPhotonPairsPath,MixedR9CaloIdPhotonPairsPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

########## B-Parking #########
from Configuration.Skimming.pwdgSkimBPark_cfi import *
SkimBParkPath = cms.Path(SkimBPark)
SKIMStreamSkimBPark = cms.FilteredStream(
    responsible = 'BPH PAG',
    name = 'SkimBPark',
    paths = ( SkimBParkPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
)

from Configuration.EventContent.EventContent_cff import AODEventContent
skimAodContent = AODEventContent.clone()
skimAodContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimAodContent.outputCommands.append("drop *_*_*_SKIM")

#from Configuration.Skimming.PDWG_DoublePhotonSkim_cff import *
#diphotonSkimPath = cms.Path(diphotonSkimSequence)
#SKIMStreamDoublePhoton = cms.FilteredStream(
#    responsible = 'PDWG',
#    name = 'DoublePhoton',
#    paths = (diphotonSkimPath),
#    content = skimAodContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('AOD')
#    )

from Configuration.Skimming.PDWG_HWWSkim_cff import *
HWWmmPath = cms.Path(diMuonSequence)
HWWeePath = cms.Path(diElectronSequence)
HWWemPath = cms.Path(EleMuSequence)
SKIMStreamHWW = cms.FilteredStream(
        responsible = 'PDWG',
        name = 'HWW',
        paths = (HWWmmPath,HWWeePath,HWWemPath),
        content = skimAodContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('AOD')
        )


from Configuration.Skimming.PDWG_HZZSkim_cff import *
HZZmmPath = cms.Path(zzdiMuonSequence)
HZZeePath = cms.Path(zzdiElectronSequence)
HZZemPath = cms.Path(zzeleMuSequence)
SKIMStreamHZZ = cms.FilteredStream(
        responsible = 'PDWG',
        name = 'HZZ',
        paths = (HZZmmPath,HZZeePath,HZZemPath),
        content = skimAodContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('AOD')
        )


from Configuration.Skimming.PDWG_EXOHPTE_cff import *
exoHPTEPath = cms.Path(exoDiHPTESequence)
SKIMStreamEXOHPTE = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXOHPTE',
    paths = (exoHPTEPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from Configuration.Skimming.PDWG_EXOMONOPOLE_cff import *
EXOMONOPOLEPath = cms.Path(EXOMonopoleSkimSequence)
SKIMStreamEXOMONOPOLE = cms.FilteredStream(
        responsible = 'PDWG',
        name = 'EXOMONOPOLE',
        paths = (EXOMONOPOLEPath),
        content = EXOMonopoleSkimContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('USER')
        )

from Configuration.Skimming.PDWG_EXOHighMET_cff import *
EXOHighMETPath = cms.Path(EXOHighMETSequence)
SKIMStreamEXOHighMET = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXOHighMET',
    paths = (EXOHighMETPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

from Configuration.Skimming.PDWG_EXODisplacedJet_cff import *
EXODisplacedJetPath = cms.Path(EXODisplacedJetSkimSequence)
SKIMStreamEXODisplacedJet = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXODisplacedJet',
    paths = (EXODisplacedJetPath),
    content = skimRawAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

from Configuration.Skimming.PDWG_EXODelayedJet_cff import *
EXODelayedJetPath = cms.Path(EXODelayedJetSkimSequence)
SKIMStreamEXODelayedJet = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXODelayedJet',
    paths = (EXODelayedJetPath),
    content = skimRawAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from Configuration.Skimming.PDWG_EXODelayedJetMET_cff import *
EXODelayedJetMETPath = cms.Path(EXODelayedJetMETSkimSequence)
SKIMStreamEXODelayedJetMET = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXODelayedJetMET',
    paths = (EXODelayedJetMETPath),
    content = skimRawAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from Configuration.Skimming.PDWG_EXODTCluster_cff import *
EXODTClusterPath = cms.Path(EXODTClusterSkimSequence)
SKIMStreamEXODTCluster = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXODTCluster',
    paths = (EXODTClusterPath),
    content = skimRawAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from Configuration.Skimming.PDWG_EXOCSCCluster_cff import *
EXOCSCClusterPath = cms.Path(EXOCSCClusterSkimSequence)
SKIMStreamEXOCSCCluster = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXOCSCCluster',
    paths = (EXOCSCClusterPath),
    content = skimRawAODContent.outputCommands+['keep *_csc2DRecHits_*_*','keep *_dt1DRecHits_*_*'],
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )


from Configuration.Skimming.PDWG_EXODisappTrk_cff import *
EXODisappTrkPath = cms.Path(EXODisappTrkSkimSequence)
SKIMStreamEXODisappTrk = cms.FilteredStream(
    responsible = 'PDWG', 
    name = 'EXODisappTrk', 
    paths = (EXODisappTrkPath),
    content = EXODisappTrkSkimContent.outputCommands, 
    selectEvents = cms.untracked.PSet(), 
    dataTier = cms.untracked.string('AOD')
    )

#####################
# For the Data on Data Mixing in TSG
from HLTrigger.Configuration.HLT_Fake1_cff import fragment as _fragment
if "hltGtDigis" in _fragment.__dict__:
    hltGtDigis = _fragment.hltGtDigis.clone()
    hltGtDigisPath = cms.Path(hltGtDigis)
else:
    hltBoolEnd = _fragmet.hltBoolEnd.clone()
    hltGtDigisPath = cms.Path(hltBoolEnd)

# The events to be used as PileUp
from Configuration.Skimming.PDWG_HLTZEROBIASPU_SD_cff import *
HLTZEROBIASPUSDPath = cms.Path(HLTZEROBIASPUSD)
SKIMStreamHLTZEROBIASPUSD = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'HLTZEROBIASPUSD',
    paths = (HLTZEROBIASPUSDPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW') # for the moment, it could be DIGI in the future
    )

#The events to be used as signal
from Configuration.Skimming.PDWG_HLTZEROBIASSIG_SD_cff import *
HLTZEROBIASSIGSDPath = cms.Path(HLTZEROBIASSIGSD)
SKIMStreamHLTZEROBIASSIGSD = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'HLTZEROBIASSIGSD',
    paths = (HLTZEROBIASSIGSDPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW') # for the moment, it could be DIGI in the future
    )

####################
   


## exo skims
"""
from SUSYBSMAnalysis.Skimming.EXOLLResSkim_cff import *
exoLLResmmPath = cms.Path(exoLLResdiMuonSequence)
exoLLReseePath = cms.Path(exoLLResdiElectronSequence)
exoLLResemPath = cms.Path(exoLLResEleMuSequence)
SKIMStreamEXOLLRes = cms.FilteredStream(
        responsible = 'EXO',
        name = 'EXOLLRes',
        paths = (exoLLResmmPath,exoLLReseePath,exoLLResemPath),
        content = skimAodContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('AOD')
        )

from SUSYBSMAnalysis.Skimming.EXOEle_cff import *
exoElePath = cms.Path(exoEleLowetSeqReco)
SKIMStreamEXOEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOEle',
    paths = (exoElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXOMu_cff import *
exoMuPath = cms.Path(exoMuSequence)
SKIMStreamEXOMu = cms.FilteredStream(
    responsible = 'EXO',
    name = "EXOMu",
    paths = (exoMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXOTriLepton_cff import *
exoTriMuPath = cms.Path(exoTriMuonSequence)
SKIMStreamEXOTriMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOTriMu',
    paths = (exoTriMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
exoTriElePath = cms.Path(exoTriElectronSequence)
SKIMStreamEXOTriEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOTriEle',
    paths = (exoTriElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
exo1E2MuPath = cms.Path(exo1E2MuSequence)
SKIMStreamEXO1E2Mu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXO1E2Mu',
    paths = (exo1E2MuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXODiLepton_cff import *
exoDiMuPath = cms.Path(exoDiMuSequence)
exoDiElePath = cms.Path(exoDiMuSequence)
exoEMuPath = cms.Path(exoEMuSequence)
SKIMStreamEXODiMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXODiMu',
    paths = (exoDiMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
SKIMStreamEXODiEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXODiEle',
    paths = (exoDiElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
SKIMStreamEXOEMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOEMu',
    paths = (exoEMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
"""

#####################

from Configuration.Skimming.PDWG_MuonPOGSkim_cff import *
MuonPOGSkimTrackPath = cms.Path(MuonPOGSkimTrackSequence)
MuonPOGSkimSTAPath   = cms.Path(MuonPOGSkimSTASequence)
SKIMStreamMuonPOGSkim     = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'MuonPOGSkim',
    paths = (MuonPOGSkimTrackPath,MuonPOGSkimSTAPath),
    content = MuonPOG_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )


MuonPOGJPsiSkimTrackPath = cms.Path(MuonPOGJPsiSkimTrackSequence)
MuonPOGJPsiSkimSTAPath   = cms.Path(MuonPOGJPsiSkimSTASequence)

SKIMStreamMuonPOGJPsiSkim     = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'MuonPOGJPsiSkim',
    paths = (MuonPOGJPsiSkimTrackPath,MuonPOGJPsiSkimSTAPath),
    content = MuonPOG_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )


#####################
