# The following comments couldn't be translated into the new config version:

# of sigma above threshold for a cell to be considered hot
# orig vale was 0.02
# HotCell Tags for individual subdetectors
# If these aren't specified, then the global 
# values are used
#untracked bool HBdebug = false
#untracked bool HEdebug = false
#untracked bool HOdebug = false
#untracked bool HFdebug = false
# Re-tuned threshold based on CRUZET1 run 43636 
# First threshold is used when looking for hot cells 

import FWCore.ParameterSet.Config as cms

hcalMonitor = cms.EDFilter("HcalMonitorModule",

                           # GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different values can trigger different levels of messaging
                           
                           MaxEta = cms.untracked.double(44.5),
                           MinEta = cms.untracked.double(-44.5),
                           MaxPhi = cms.untracked.double(73.5),
                           MinPhi = cms.untracked.double(-0.5),

                           checkHF = cms.untracked.bool(True),
                           checkHE = cms.untracked.bool(True),
                           checkHB = cms.untracked.bool(True),
                           checkHO = cms.untracked.bool(True),

                           minErrorFlag = cms.untracked.double(0.05), # minimum error rate that will cause problem cells to be filled
                           
                           DumpThreshold = cms.untracked.double(500.0),
                           thresholds = cms.untracked.vdouble(15.0, 5.0, 2.0, 1.5, 1.0),
                           coolcellfrac = cms.untracked.double(0.5),
                           NADA_maxeta = cms.untracked.int32(1),
                           hoRecHitLabel = cms.InputTag("horeco"),
                           MakeHotCellDiagnosticPlots = cms.untracked.bool(False),
                           DigiMonitor = cms.untracked.bool(True),
                           HotCell_checkNADA = cms.untracked.bool(True),
                           digiLabel = cms.InputTag("hcalDigis"),
                           hfRecHitLabel = cms.InputTag("hfreco"),
                           zdcRecHitLabel = cms.InputTag("zdcreco"),                           
                           hcalLaserLabel = cms.InputTag("hcalLaserReco"),                       
                           DumpPhiLow = cms.untracked.int32(10),
                           DigiOccThresh = cms.untracked.int32(0),
                           RecHitsPerChannel = cms.untracked.bool(False),

                           # PEDESTAL MONITOR
                           PedestalMonitor                              = cms.untracked.bool(True),
                           PedestalMonitor_pedestalsPerChannel          = cms.untracked.bool(True), # not used
                           PedestalMonitor_pedestalsInFC                = cms.untracked.bool(False),
                           PedestalMonitor_nominalPedMeanInADC          = cms.untracked.double(3.),
                           PedestalMonitor_nominalPedWidthInADC         = cms.untracked.double(1.),
                           PedestalMonitor_maxPedMeanDiffADC            = cms.untracked.double(1.),
                           PedestalMonitor_maxPedWidthDiffADC           = cms.untracked.double(1.),
                           PedestalMonitor_startingTimeSlice            = cms.untracked.int32(0),
                           PedestalMonitor_endingTimeSlice              = cms.untracked.int32(1),
                           PedestalMonitor_minErrorFlag                 = cms.untracked.double(0.05),
                           PedestalMonitor_checkNevents                 = cms.untracked.int32(100),

                           
                           HE_NADA_Ecell_cut = cms.untracked.double(0.0),
                           HF_NADA_Ecube_frac = cms.untracked.double(0.5714),
                           HB_NADA_Ecell_frac = cms.untracked.double(0.0),
                           HF_NADA_Ecand_cut2 = cms.untracked.double(20.0),
                           HF_NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           HF_NADA_Ecand_cut1 = cms.untracked.double(1.5),
                           diagnosticPrescaleLS = cms.untracked.int32(-1),
                           MakeDigiDiagnosticPlots = cms.untracked.bool(False),
                           CaloTowerMonitor = cms.untracked.bool(False),
                           BeamMonitor = cms.untracked.bool(True),
                           ExpertMonitor = cms.untracked.bool(False),
                           NADA_Ecube_frac = cms.untracked.double(0.3),
                           NADA_maxphi = cms.untracked.int32(1),
                           minEntriesPerPed = cms.untracked.int32(10),
                           RecHitOccThresh = cms.untracked.double(2.0),
                           NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           NADA_Ecand_cut1 = cms.untracked.double(2.5),
                           NADA_Ecand_cut2 = cms.untracked.double(500.0),
                           MonitorDaemon = cms.untracked.bool(True),
                           HE_NADA_Ecand_cut1 = cms.untracked.double(2.5),
                           RecHitMonitor = cms.untracked.bool(True),
                           caloTowerLabel = cms.InputTag("towerMaker"),
                           HE_NADA_Ecand_cut2 = cms.untracked.double(10.0),
                           HF_NADA_Ecell_cut = cms.untracked.double(0.0),
                           MakeDiagnosticPlots = cms.untracked.bool(True),
                           NADA_Ecell_frac = cms.untracked.double(0.02),
                           HO_NADA_Ecube_cut = cms.untracked.double(0.1),
                           HF_NADA_Ecell_frac = cms.untracked.double(0.0),
                           HB_NADA_Ecell_cut = cms.untracked.double(0.0),
                           MTCCMonitor = cms.untracked.bool(False),
                           HcalAnalysis = cms.untracked.bool(False),
                           DumpEtaHigh = cms.untracked.int32(10),
                           DigisPerChannel = cms.untracked.bool(False),
                           HEthresholds = cms.untracked.vdouble(3.0, 2.0, 1.0, 5.0, 10.0),
                           LED_ADC_Thresh = cms.untracked.double(-1000.0),
                           diagnosticPrescaleTime = cms.untracked.int32(-1),
                           LEDPerChannel = cms.untracked.bool(True),

                           NADA_NegCand_cut = cms.untracked.double(-1.5),
                           HE_NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           NADA_maxdepth = cms.untracked.int32(0),
                           hbheRecHitLabel = cms.InputTag("hbhereco"),
                           HotCellDigiSigma = cms.untracked.double(5.0),
                           DataFormatMonitor = cms.untracked.bool(True),
                           DataIntegrityTask = cms.untracked.bool(False),
                           HFthresholds = cms.untracked.vdouble(9.0, 2.0, 4.0, 6.0, 10.0),
                           HotCell_checkThreshold = cms.untracked.bool(True),
                           minADCcount = cms.untracked.double(0.0),

                           # maximum allowed diff between observed, db pedestals before error is thrown
                           maxPedestalDiffADC=cms.untracked.double(1.),
                           maxPedestalWidthADC=cms.untracked.double(2.),
                           
                           HO_NADA_Ecell_frac = cms.untracked.double(0.0),
                           HE_NADA_Ecell_frac = cms.untracked.double(0.0),
                           showTiming = cms.untracked.bool(False),
                           makeSubdetHistos= cms.untracked.bool(True),
                           HB_NADA_Ecube_frac = cms.untracked.double(0.333),
                           HB_NADA_Ecand_cut2 = cms.untracked.double(10.0),
                           HB_NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           HB_NADA_Ecand_cut1 = cms.untracked.double(2.25),
                           HF_NADA_Ecube_cut = cms.untracked.double(0.1),
                           HOthresholds = cms.untracked.vdouble(2.0, 3.0, 1.0, 5.0, 10.0),
                           ped_Nsigma = cms.untracked.double(-3.1),
                           HotCells = cms.untracked.vstring(),
                           DeadCellMonitor = cms.untracked.bool(True),
                           HO_NADA_Ecand_cut2 = cms.untracked.double(10.0),
                           HO_NADA_Ecand_cut1 = cms.untracked.double(2.5),
                           HO_NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           NADA_Ecell_cut = cms.untracked.double(0.1),
                           diagnosticPrescaleEvt = cms.untracked.int32(-1),
                           HE_NADA_Ecube_frac = cms.untracked.double(0.222),
                           DumpPhiHigh = cms.untracked.int32(10),
                           HB_NADA_Ecube_cut = cms.untracked.double(0.1),
                           DeadCell_checkAbovePed = cms.untracked.bool(True),
                           gtLabel = cms.InputTag("l1GtUnpack"),
                           HO_NADA_Ecell_cut = cms.untracked.double(0.0),
                           DumpEtaLow = cms.untracked.int32(0),
                           HotCell_checkAbovePed = cms.untracked.bool(True),
                           TrigPrimMonitor = cms.untracked.bool(True),
                           
                           MakeDeadCellDiagnosticPlots = cms.untracked.bool(False),
                           HotCellMonitor = cms.untracked.bool(True),
                           deadcellmindiff = cms.untracked.double(1.0),
                           checkNevents = cms.untracked.int32(250),
                           HBcheckNevents = cms.untracked.int32(250),
                           HEcheckNevents = cms.untracked.int32(250),
                           HOcheckNevents = cms.untracked.int32(1000),
                           HFcheckNevents = cms.untracked.int32(250),
                           
                           diagnosticPrescaleUpdate = cms.untracked.int32(-1),
                           HE_NADA_Ecube_cut = cms.untracked.double(0.1),
                           HO_NADA_Ecube_frac = cms.untracked.double(0.2),
                           
                           MTCCOccThresh = cms.untracked.double(10.0),
                           NADA_Ecube_cut = cms.untracked.double(0.5),
                           HBthresholds = cms.untracked.vdouble(3.0, 2.0, 1.0, 5.0, 10.0),
                           LEDMonitor = cms.untracked.bool(True),
                           
                           deadcellfloor = cms.untracked.double(-10.0),
                           dfPrtLvl = cms.untracked.int32(0)
                           )


def setHcalTaskValues(process):
    # If you import this function directly, you can then set all the individual subtask values to the global settings
    # (This is useful if you've changed the global value, and you want it to propagate everywhere)

    # Set minimum value needed to put an entry into Problem histograms.  (values are between 0-1)
    minErrorFlags = [process.PedestalMonitor_minErrorFlag]
    for i in range(len(minErrorFlags)):
        minErrorFlags[i] = process.minErrorFlag

    checkNevents = [process.PedestalMonitor_checkNevents]
    for i in range(len(checkNevents)):
        checkNevents[i] = process.checkNevents

    return
