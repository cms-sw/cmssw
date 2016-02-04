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

SKIMStreamDiPhoton = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiPhoton',
    paths = (CaloIdIsoPhotonPairsPath,R9IdPhotonPairsPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


from Configuration.EventContent.EventContent_cff import AODEventContent
skimAodContent = AODEventContent.clone()
skimAodContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimAodContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.Skimming.PDWG_DoublePhotonSkim_cff import *
diphotonSkimPath = cms.Path(diphotonSkimSequence)
SKIMStreamDoublePhoton = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DoublePhoton',
    paths = (diphotonSkimPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

#from Configuration.Skimming.PDWG_EXOHSCP_cff import *
#EXOHSCPPath = cms.Path(exoticaHSCPSeq)
#SKIMStreamEXOHSCP = cms.FilteredStream(
#    responsible = 'PDWG',
#    name = 'EXOHSCP',
#    paths = (EXOHSCPPath),
#    content = EXOHSCPSkim_EventContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('USER')
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
