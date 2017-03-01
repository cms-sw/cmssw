
import FWCore.ParameterSet.Config as cms

def reEmulateLayer2(process):

    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')
    process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_HI_cfi')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    
    process.caloLayer2 = cms.Path(process.simCaloStage2Digis)

    process.schedule.append(process.caloLayer2)
   
    return process


def hwEmulCompHistos(process):

    # histograms
    process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
    process.l1tStage2CaloAnalyzer.doEvtDisp = False
    process.l1tStage2CaloAnalyzer.mpBx = 0
    process.l1tStage2CaloAnalyzer.dmxBx = 0
    process.l1tStage2CaloAnalyzer.allBx = False
    process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("simCaloStage2Digis", "MP")
    process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
    process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("simCaloStage2Digis", "MP")
    process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("simCaloStage2Digis", "MP")
    process.l1tStage2CaloAnalyzer.mpJetToken = cms.InputTag("simCaloStage2Digis", "MP")
    process.l1tStage2CaloAnalyzer.mpEtSumToken = cms.InputTag("simCaloStage2Digis", "MP")
    process.l1tStage2CaloAnalyzer.egToken = cms.InputTag("simCaloStage2Digis")
    process.l1tStage2CaloAnalyzer.tauToken = cms.InputTag("simCaloStage2Digis")
    process.l1tStage2CaloAnalyzer.jetToken = cms.InputTag("simCaloStage2Digis")
    process.l1tStage2CaloAnalyzer.etSumToken = cms.InputTag("simCaloStage2Digis")
    
    import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
    process.l1tCaloStage2HwHistos =  L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
    process.l1tCaloStage2HwHistos.doEvtDisp = False
    process.l1tCaloStage2HwHistos.mpBx = 0
    process.l1tCaloStage2HwHistos.dmxBx = 0
    process.l1tCaloStage2HwHistos.allBx = False
    process.l1tCaloStage2HwHistos.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    process.l1tCaloStage2HwHistos.clusterToken = cms.InputTag("None")
    process.l1tCaloStage2HwHistos.mpEGToken = cms.InputTag("caloStage2Digis", "MP")
    process.l1tCaloStage2HwHistos.mpTauToken = cms.InputTag("caloStage2Digis","MP")
    process.l1tCaloStage2HwHistos.mpJetToken = cms.InputTag("caloStage2Digis", "MP")
    process.l1tCaloStage2HwHistos.mpEtSumToken = cms.InputTag("caloStage2Digis", "MP")
    process.l1tCaloStage2HwHistos.egToken = cms.InputTag("caloStage2Digis", "EGamma")
    process.l1tCaloStage2HwHistos.tauToken = cms.InputTag("caloStage2Digis", "Tau")
    process.l1tCaloStage2HwHistos.jetToken = cms.InputTag("caloStage2Digis", "Jet")
    process.l1tCaloStage2HwHistos.etSumToken = cms.InputTag("caloStage2Digis", "EtSum")

    process.hwEmulHistos = cms.Path(
        process.l1tStage2CaloAnalyzer
        +process.l1tCaloStage2HwHistos
    )

    process.schedule.append(process.hwEmulHistos)

    return process


def reEmulateLayer2ValHistos(process):

    process.load('EventFilter.L1TRawToDigi.caloTowersFilter_cfi')

    reEmulateLayer2(process)
    hwEmulCompHistos(process)

    process.l1ntupleraw.insert(0,process.caloTowersFilter)
    #process.l1ntuplesim.insert(0,process.caloTowersFilter)
    process.caloLayer2.insert(0,process.caloTowersFilter)
    process.hwEmulHistos.insert(0,process.caloTowersFilter)

    return process

