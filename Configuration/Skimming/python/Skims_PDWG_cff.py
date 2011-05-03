import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.Skims_DPG_cff import skimContent

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
tauSkimPath = cms.Path( tauSkimSequence )
SKIMStreamTau = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Tau',
    paths = (tauSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


#####################

from Configuration.Skimming.PDWG_OniaSkim_cff import *
oniaSkimPath = cms.Path(oniaSkimSequence)
SKIMStreamOnia = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Onia',
    paths = (oniaSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from Configuration.Skimming.PDWG_HT_SD_cff import *
HTSDPath = cms.Path(HTSD)
SKIMStreamHTSD = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'HTSD',
    paths = (HTSDPath),
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
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from Configuration.EventContent.EventContent_cff import RECOEventContent
skimRecoContent = RECOEventContent.clone()
skimRecoContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRecoContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.Skimming.PDWG_SuperClusterSkim_cff import *
diSuperClusterSkimPath = cms.Path(diSuperClusterSkimSequence)
SKIMStreamSuperCluster = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'SuperCluster',
    paths = (diSuperClusterSkimPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

from Configuration.Skimming.PDWG_LeptonSkim_cff import *
SingleMuPath = cms.Path(filterSingleMu)
DoubleMuPath = cms.Path(filterDoubleMu)
MuElectronPath = cms.Path(filterMuonElectron)
MuPFElectronPath = cms.Path(filterMuonPFElectron)
DoubleElectronPath = cms.Path(filterDoubleElectron)
DoublePFElectronPath = cms.Path(filterDoublePFElectron)
SKIMStreamDiLeptonMu = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiLeptonMu',
    paths = (DoubleMuPath,MuElectronPath,MuPFElectronPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )
SKIMStreamSingleMu = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'SingleMu',
    paths = (SingleMuPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )
SKIMStreamDiLeptonEle = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiLeptonEle',
    paths = (DoubleElectronPath,DoublePFElectronPath,MuElectronPath,MuPFElectronPath),
    content = skimRecoContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

from Configuration.Skimming.PDWG_DiPhoton_SD_cff import *
CaloIdIsoPhotonPairsPath = cms.Path(CaloIdIsoPhotonPairsFilter)
R9IdPhotonPairsPath = cms.Path(R9IdPhotonPairsFilter)

SKIMStreamDiPhoton = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiPhoton',
    paths = (CaloIdIsoPhotonPairsPath,R9IdPhotonPairsPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

