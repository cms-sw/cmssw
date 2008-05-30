import FWCore.ParameterSet.Config as cms

# The purpose of this file is to save data into separate
# files (primary datasets) based on which HLT trigger bits are set.
# Each PoolOutputModule defines a file and HLT filter paths to
# select the event from.
from Configuration.EventContent.EventContent_cff import *
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdTauFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdPhotonFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdMuonFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdElectronFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdBJetFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
pdJetMETFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
HLTPrimaryDatasetsDefaults = cms.PSet(
    basketSize = cms.untracked.int32(4096),
    outputCommands = cms.untracked.vstring('drop *')
)
HLTPrimaryDatasetsDefaultPSet = cms.PSet(
    dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
)
CSA07Tau = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07Tau.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdTauPath')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07Tau')
    )
)

CSA07Photon = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07Photon.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdPhotonPath')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07Photon')
    )
)

CSA07Muon = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07Muon.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdMuonPath')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07Muon')
    )
)

CSA07Electron = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07Electron.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdElectronPath')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07Electron')
    )
)

CSA07BJet = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07BJet.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdBJet')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07BJet')
    )
)

CSA07JetMET = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07JetMET.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdJetMETPath')
    ),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07JetMET')
    )
)

CSA07AllEvents = cms.OutputModule("PoolOutputModule",
    HLTPrimaryDatasetsDefaults,
    fileName = cms.untracked.string('file:CSA07AllEvents.root'),
    dataset = cms.untracked.PSet(
        HLTPrimaryDatasetsDefaultPSet,
        filterName = cms.untracked.string('CSA07AllEvents')
    )
)

pdTauPath = cms.Path(pdTauFilter)
pdPhotonPath = cms.Path(pdPhotonFilter)
pdMuonPath = cms.Path(pdMuonFilter)
pdElectronPath = cms.Path(pdElectronFilter)
pdBJet = cms.Path(pdBJetFilter)
pdJetMETPath = cms.Path(pdJetMETFilter)
eCSA07Tau = cms.EndPath(CSA07Tau)
eCSA07Photon = cms.EndPath(CSA07Photon)
eCSA07Muon = cms.EndPath(CSA07Muon)
eCSA07Electron = cms.EndPath(CSA07Electron)
eCSA07BJet = cms.EndPath(CSA07BJet)
eCSA07JetMET = cms.EndPath(CSA07JetMET)
eCSA07AllEvents = cms.EndPath(CSA07AllEvents)
HLTPrimaryDatasetsDefaults.outputCommands.extend(FEVTSIMEventContent.outputCommands)
HLTPrimaryDatasetsDefaults.outputCommands.append('drop *_*Digis_*_HLT')
HLTPrimaryDatasetsDefaults.outputCommands.append('drop recoIsolatedTauTagInfos_*_*_*')
pdTauFilter.HLTPaths = ['HLT1Tau', 'HLT1Tau1MET', 'HLT2TauPixel', 'HLTXMuonTau', 'HLTXElectronTau']
pdPhotonFilter.HLTPaths = ['HLT1Photon', 'HLT1PhotonRelaxed', 'HLT2Photon', 'HLT2PhotonRelaxed', 'HLT2PhotonExclusive']
pdMuonFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLT2MuonJPsi', 'HLT2MuonUpsilon', 
    'HLT2MuonZ', 'HLTNMuonNonIso', 'HLT2MuonSameSign', 'HLTBJPsiMuMu', 'HLTXMuonJets', 
    'HLTXElectronMuon', 'HLTXElectronMuonRelaxed', 'HLTXMuonTau']
pdElectronFilter.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed', 'HLT2Electron', 'HLT2ElectronRelaxed', 'HLT1EMHighEt', 
    'HLT1EMVeryHighEt', 'HLTXElectron1Jet', 'HLTXElectron2Jet', 'HLTXElectron3Jet', 'HLTXElectron4Jet', 
    'HLTXElectronMuon', 'HLTXElectronMuonRelaxed', 'HLTXElectronTau', 'HLT2ElectronZCounter', 'HLT2ElectronExclusive']
pdBJetFilter.HLTPaths = ['HLTB2JetMu', 'HLTB3JetMu', 'HLTB4JetMu', 'HLTBHTMu', 'HLTXMuonBJet', 
    'HLTXMuonBJetSoftMuon', 'HLTB1Jet', 'HLTB2Jet', 'HLTB3Jet', 'HLTB4Jet', 
    'HLTBHT', 'HLTXElectronBJet']
pdJetMETFilter.HLTPaths = ['HLT1jet30', 'HLT1jet50', 'HLT1jet80', 'HLT1jet110', 'HLT1jet180', 
    'HLT1jet250', 'HLT2jet', 'HLT3jet', 'HLT4jet', 'HLT1MET25', 
    'HLT1MET35', 'HLT1MET50', 'HLT1MET65', 'HLT1MET75', 'HLT2jetAco', 
    'HLT1jet1METAco', 'HLT1jet1MET', 'HLT2jet1MET', 'HLT3jet1MET', 'HLT4jet1MET', 
    'HLT1MET1HT', 'HLT2jetvbfMET', 'HLTS2jet1METNV', 'HLTS2jet1METAco', 'HLTSjet1MET1Aco', 
    'HLTSjet2MET1Aco', 'HLTS2jetMET1Aco', 'HLTJetMETRapidityGap']


