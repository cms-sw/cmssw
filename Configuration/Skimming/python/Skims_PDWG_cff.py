import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.Skims_DPG_cff import skimContent

from Configuration.EventContent.EventContent_cff import RECOEventContent
skimRecoContent = RECOEventContent.clone()
skimRecoContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRecoContent.outputCommands.append("drop *_*_*_SKIM")

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

from Configuration.Skimming.PDWG_EXOHSCP_cff import *
EXOHSCPPath = cms.Path(exoticaHSCPSeq)
SKIMStreamEXOHSCP = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'EXOHSCP',
    paths = (EXOHSCPPath),
    content = EXOHSCPSkim_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

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
