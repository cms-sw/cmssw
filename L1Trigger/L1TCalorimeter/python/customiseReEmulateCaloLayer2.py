import FWCore.ParameterSet.Config as cms

def hwEmulCompHistos(process):
    
    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string("l1tCalo_2016_simHistos.root"),
                                       closeFileFast = cms.untracked.bool(True)
                                       )
    
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




def valHistosDB(process):

    process.load('EventFilter.L1TRawToDigi.caloTowersFilter_cfi')
    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')
    process.simCaloStage2Digis.useStaticConfig = False
    process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_8_4_cfi')

    process.load('L1Trigger.Configuration.SimL1Emulator_cff')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    process.caloLayer2 = cms.Path(process.simCaloStage2Digis)

    process.schedule.append(process.caloLayer2)

    hwEmulCompHistos(process)

    process.caloLayer2.insert(0,process.caloTowersFilter)
    process.hwEmulHistos.insert(0,process.caloTowersFilter)

    return process



def valHistosStatic(process):

    process.load('EventFilter.L1TRawToDigi.caloTowersFilter_cfi')
    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')
    process.simCaloStage2Digis.useStaticConfig = True
    process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_8_4_cfi')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    process.caloLayer2 = cms.Path(process.simCaloStage2Digis)

    process.schedule.append(process.caloLayer2)

    hwEmulCompHistos(process)

    process.caloLayer2.insert(0,process.caloTowersFilter)
    process.hwEmulHistos.insert(0,process.caloTowersFilter)

    return process


def L1NtupleRAWEMU(process):

    process.load('L1Trigger.L1TNtuples.L1NtupleRAW_cff')
    process.load('L1Trigger.L1TNtuples.L1NtupleEMU_cff')

    process.l1ntuplerawemu = cms.Path( process.L1NtupleRAW
                                    + process.L1NtupleEMU )
    process.schedule.append(process.l1ntuplerawemu)
    
    return process


def valHistosDBL1Ntuple(process):

    process.load('EventFilter.L1TRawToDigi.caloTowersFilter_cfi')
    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')
    process.simCaloStage2Digis.useStaticConfig = False
    process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_8_4_cfi')

    process.load('L1Trigger.Configuration.SimL1Emulator_cff')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    process.caloLayer2 = cms.Path(process.simCaloStage2Digis)

    process.schedule.append(process.caloLayer2)

    hwEmulCompHistos(process)
    L1NtupleRAWEMU(process)

    process.caloLayer2.insert(0,process.caloTowersFilter)
    process.hwEmulHistos.insert(0,process.caloTowersFilter)
    process.l1ntuplerawemu.insert(0,process.caloTowersFilter)

    return process



def valHistosStaticL1Ntuple(process):

    process.load('EventFilter.L1TRawToDigi.caloTowersFilter_cfi')
    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')
    process.simCaloStage2Digis.useStaticConfig = True
    process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_8_4_cfi')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")
    process.caloLayer2 = cms.Path(process.simCaloStage2Digis)

    process.schedule.append(process.caloLayer2)

    hwEmulCompHistos(process)
    L1NtupleRAWEMU(process)

    process.caloLayer2.insert(0,process.caloTowersFilter)
    process.hwEmulHistos.insert(0,process.caloTowersFilter)
    process.l1ntuplerawemu.insert(0,process.caloTowersFilter)

    return process




