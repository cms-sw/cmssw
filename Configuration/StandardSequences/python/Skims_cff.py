import FWCore.ParameterSet.Config as cms

def documentSkims():
    import Configuration.StandardSequences.Skims_cff as Skims

    listOfOptions=[]
    for skim in Skims.__dict__:
        skimstream = getattr(Skims,skim)
        if (not isinstance(skimstream,cms.FilteredStream)):
            continue
        
        shortname = skim.replace('SKIMStream','')
        print shortname
        if shortname!=skimstream['name']:
            print '#### ERROR ####'
            print 'skim name and stream name should be the same for consistency',shortname,'!=',skimstream['name']
            
        for token in ['name','responsible','dataTier']:
            print token,":",skimstream[token]
            
        listOfOptions.append(skimstream['name'])

    print 'possible cmsDriver options for skimming:'
    print 'SKIM:'+'+'.join(listOfOptions)


### DPG skims ###
from DPGAnalysis.Skims.Skims_DPG_cff import *


### Central Skims ###
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

