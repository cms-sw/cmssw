import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.WprimeMonitor_cfi import hltWprimemonitoring

WprimeEle115 = hltWprimemonitoring.clone()
WprimeEle115.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle115')
WprimeEle115.nmuons = cms.uint32(0)
WprimeEle115.nelectrons = cms.uint32(1)
WprimeEle115.njets = cms.uint32(0)
WprimeEle115.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle115.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle115.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele115_CaloIdVT_GsfTrkIdT_v*')
WprimeEle115.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )



WprimeEle135 = hltWprimemonitoring.clone()
WprimeEle135.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle135')
WprimeEle135.nmuons = cms.uint32(0)
WprimeEle135.nelectrons = cms.uint32(1)
WprimeEle135.njets = cms.uint32(0)
WprimeEle135.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle135.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')

WprimeEle135.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele135_CaloIdVT_GsfTrkIdT_v*')
WprimeEle135.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )



WprimeEle145 = hltWprimemonitoring.clone()
WprimeEle145.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle145')
WprimeEle145.nmuons = cms.uint32(0)
WprimeEle145.nelectrons = cms.uint32(1)
WprimeEle145.njets = cms.uint32(0)
WprimeEle145.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle145.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')

WprimeEle145.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele145_CaloIdVT_GsfTrkIdT_v*')
WprimeEle145.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )

WprimeEle200 = hltWprimemonitoring.clone()
WprimeEle200.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle200')
WprimeEle200.nmuons = cms.uint32(0)
WprimeEle200.nelectrons = cms.uint32(1)
WprimeEle200.njets = cms.uint32(0)
WprimeEle200.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle200.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')

WprimeEle200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele200_CaloIdVT_GsfTrkIdT_v*')
WprimeEle200.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )

WprimeEle250 = hltWprimemonitoring.clone()
WprimeEle250.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle250')
WprimeEle250.nmuons = cms.uint32(0)
WprimeEle250.nelectrons = cms.uint32(1)
WprimeEle250.njets = cms.uint32(0)
WprimeEle250.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle250.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')

WprimeEle250.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele250_CaloIdVT_GsfTrkIdT_v*')
WprimeEle250.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )

WprimeEle300 = hltWprimemonitoring.clone()
WprimeEle300.FolderName = cms.string('HLT/EXO/Wprime/WprimeEle300')
WprimeEle300.nmuons = cms.uint32(0)
WprimeEle300.nelectrons = cms.uint32(1)
WprimeEle300.njets = cms.uint32(0)
WprimeEle300.eleSelection = cms.string('pt>50 & abs(eta)<1.4442 & full5x5_sigmaIetaIeta<0.00998 & abs(deltaEtaSuperClusterAtVtx)<0.00308 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0816 & hadronicOverEm<0.0414 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
WprimeEle300.eleSelection1 = cms.string('pt>50 & abs(eta)>1.566 & abs(eta)<2.5 & full5x5_sigmaIetaIeta<0.0292 & abs(deltaEtaSuperClusterAtVtx)<0.00605 & abs(deltaPhiSuperClusterTrackAtVtx)< 0.0394 & hadronicOverEm<0.0641 & abs(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)<0.0129 & passConversionVeto==1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')

WprimeEle300.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele300_CaloIdVT_GsfTrkIdT_v*')
WprimeEle300.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu27_v*',
                                                               'HLT_Mu50_v*',
                                                               )


WprimeMonitorHLT = cms.Sequence(
    WprimeEle115
    + WprimeEle135
    + WprimeEle145
    + WprimeEle200
    + WprimeEle250
    + WprimeEle300

    )
