# The following comments couldn't be translated into the new config version:

#List of known hot cells to veto using DetId

import FWCore.ParameterSet.Config as cms

hcalMonitor = cms.EDFilter("HcalMonitorModule",
    PedestalsPerChannel = cms.untracked.bool(True),
    NADA_NegCand_cut = cms.untracked.double(-1.5),
    RecHitOccThresh = cms.untracked.double(2.0),
    # NADA params updated for Jan DQM challenge using GREN data
    NADA_Ecand_cut0 = cms.untracked.double(1.5),
    NADA_Ecand_cut1 = cms.untracked.double(2.5),
    NADA_Ecand_cut2 = cms.untracked.double(500.0),
    checkHF = cms.untracked.bool(True),
    thresholds = cms.untracked.vdouble(1.0, 1.5, 2.0, 5.0, 15.0),
    #Determines connection to back-end daemon
    MonitorDaemon = cms.untracked.bool(True),
    NADA_maxeta = cms.untracked.int32(1),
    NADA_maxdepth = cms.untracked.int32(0),
    gtLabel = cms.InputTag("l1GtUnpack"),
    hoRecHitLabel = cms.InputTag("horeco"),
    caloTowerLabel = cms.InputTag("towerMaker"),
    hbheRecHitLabel = cms.InputTag("hbhereco"),
    #Flags for pedestal monitor
    PedestalMonitor = cms.untracked.bool(True),
    #Flags for RecHitMonitor.  Threshold is in GeV
    RecHitMonitor = cms.untracked.bool(True),
    #Flags for DigiMonitor.  Threshold is in ADC counts
    DigiMonitor = cms.untracked.bool(True),
    MinPhi = cms.untracked.double(-0.5),
    DumpEtaHigh = cms.untracked.int32(10),
    LEDPerChannel = cms.untracked.bool(True),
    DumpEtaLow = cms.untracked.int32(0),
    DumpPhiHigh = cms.untracked.int32(10),
    DumpThreshold = cms.untracked.double(500.0),
    NADA_Ecell_frac = cms.untracked.double(0.02),
    #Flags for DataFormatMonitor
    DataFormatMonitor = cms.untracked.bool(True),
    #Flags for HotCellMonitor.  Thresholds are in GeV
    HotCellMonitor = cms.untracked.bool(True),
    deadcellmindiff = cms.untracked.double(1.0),
    # Labels for input products
    digiLabel = cms.InputTag("hcalDigis"),
    checkNevents = cms.untracked.int32(25),
    minADCcount = cms.untracked.double(0.0),
    #Flags for MTCCMonitor.  Thresholds are in fC
    #Dump threshold in fC, ranges in iEta/iPhi.  Dumps to stdout
    MTCCMonitor = cms.untracked.bool(False),
    PedestalsInFC = cms.untracked.bool(False),
    # Operate every N updates
    diagnosticPrescaleUpdate = cms.untracked.int32(-1),
    #Flags for the hcal template analysis
    HcalAnalysis = cms.untracked.bool(False),
    NADA_maxphi = cms.untracked.int32(1),
    DumpPhiLow = cms.untracked.int32(10),
    checkHE = cms.untracked.bool(True),
    coolcellfrac = cms.untracked.double(0.5),
    DigiOccThresh = cms.untracked.int32(0),
    hfRecHitLabel = cms.InputTag("hfreco"),
    RecHitsPerChannel = cms.untracked.bool(False),
    checkHO = cms.untracked.bool(True),
    MaxPhi = cms.untracked.double(73.5),
    checkHB = cms.untracked.bool(True),
    DigisPerChannel = cms.untracked.bool(True),
    LED_ADC_Thresh = cms.untracked.double(-1000.0),
    # Operate every N lumi sections
    diagnosticPrescaleLS = cms.untracked.int32(-1),
    MTCCOccThresh = cms.untracked.double(10.0),
    ped_Nsigma = cms.untracked.double(0.0),
    #Flags for CaloTower Monitor
    CaloTowerMonitor = cms.untracked.bool(False),
    NADA_Ecube_cut = cms.untracked.double(0.5),
    HotCells = cms.untracked.vstring(),
    #Flags for DeadCellMonitor
    DeadCellMonitor = cms.untracked.bool(True),
    #Flags for LED monitor
    LEDMonitor = cms.untracked.bool(True),
    MinEta = cms.untracked.double(-42.5),
    NADA_Ecell_cut = cms.untracked.double(0.1),
    #Flags for TPG monitor
    TrigPrimMonitor = cms.untracked.bool(True),
    # Choices for prescaling your module (-1 mean no prescale)
    # Operate every N events
    diagnosticPrescaleEvt = cms.untracked.int32(-1),
    #Global HCAL DQM parameters.  
    #Sets range of iEta/iPhi plots
    MaxEta = cms.untracked.double(42.5),
    NADA_Ecube_frac = cms.untracked.double(0.3),
    deadcellfloor = cms.untracked.double(-10.0),
    # Operate every N minutes
    diagnosticPrescaleTime = cms.untracked.int32(-1),
    # Verbosity Switch
    debug = cms.untracked.bool(False),
    dfPrtLvl = cms.untracked.int32(0)
)


