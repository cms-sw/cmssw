import FWCore.ParameterSet.Config as cms
from copy import deepcopy

hcalMonitor = cms.EDFilter("HcalMonitorModule",

                           # GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different values can trigger different levels of messaging

                           # eta runs from -43->+43  (-41 -> +41 for HCAL, plus ZDC, which we put at |eta|=43.
                           # add one empty bin beyond that for histogramming prettiness 
                           MaxEta = cms.untracked.double(44.5),
                           MinEta = cms.untracked.double(-44.5),
                           # likewise, phi runs from 1-72.  Add some buffering bins around that region 
                           MaxPhi = cms.untracked.double(73.5),
                           MinPhi = cms.untracked.double(-0.5),

                           # Determine whether or not to check individual subdetectors
                           checkHF = cms.untracked.bool(True),
                           checkHE = cms.untracked.bool(True),
                           checkHB = cms.untracked.bool(True),
                           checkHO = cms.untracked.bool(True),

                           #minimum Error Rate that will cause problem histograms to be filled.  Should normally be 0?
                           minErrorFlag = cms.untracked.double(0.00), 

                           # Turn on/off timing diganostic info
                           showTiming = cms.untracked.bool(False),

                           # Make expert-level diagnostic plots (enabling this may drastically slow code!)
                           MakeDiagnosticPlots = cms.untracked.bool(False),

                           pedestalsInFC                               = cms.untracked.bool(False),
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
                           PedestalMonitor_checkNevents                 = cms.untracked.int32(500),
                           PedestalMonitor_minEntriesPerPed = cms.untracked.uint32(10),

                           # DEAD CELL MONITOR
                           DeadCellMonitor                              = cms.untracked.bool(True),
                           DeadCellMonitor_pedestalsInFC                = cms.untracked.bool(False),
                           DeadCellMonitor_makeDiagnosticPlots          = cms.untracked.bool(False),
                           DeadCellMonitor_test_occupancy               = cms.untracked.bool(True),
                           DeadCellMonitor_test_neighbor                = cms.untracked.bool(False), # doesn't give much useful info
                           DeadCellMonitor_test_pedestal                = cms.untracked.bool(True),
                           DeadCellMonitor_test_energy                  = cms.untracked.bool(True),
                           DeadCellMonitor_checkNevents                 = cms.untracked.int32(500),
                           DeadCellMonitor_checkNevents_occupancy       = cms.untracked.int32(500),
                           DeadCellMonitor_checkNevents_pedestal        = cms.untracked.int32(500),
                           DeadCellMonitor_checkNevents_neighbor        = cms.untracked.int32(500),
                           DeadCellMonitor_checkNevents_energy          = cms.untracked.int32(500),
                           #checking for cells consistently below (ped + Nsigma*width)
                           DeadCellMonitor_pedestal_Nsigma              = cms.untracked.double(0.),
                           DeadCellMonitor_pedestal_HB_Nsigma           = cms.untracked.double(0.),
                           DeadCellMonitor_pedestal_HE_Nsigma           = cms.untracked.double(0.),
                           DeadCellMonitor_pedestal_HO_Nsigma           = cms.untracked.double(0.),
                           DeadCellMonitor_pedestal_HF_Nsigma           = cms.untracked.double(0.),
                           # Checking for cells consistently below energy threshold
                           DeadCellMonitor_energyThreshold              = cms.untracked.double(-1.),
                           DeadCellMonitor_HB_energyThreshold           = cms.untracked.double(-1.),
                           DeadCellMonitor_HE_energyThreshold           = cms.untracked.double(-2), # |ieta=29| has many below -1.5
                           DeadCellMonitor_HO_energyThreshold           = cms.untracked.double(-3),
                           DeadCellMonitor_HF_energyThreshold           = cms.untracked.double(-1.),
                           # Check for cells below their neighbors
                           DeadCellMonitor_neighbor_deltaIeta           = cms.untracked.int32(1),
                           DeadCellMonitor_neighbor_deltaIphi           = cms.untracked.int32(1),
                           DeadCellMonitor_neighbor_deltaDepth          = cms.untracked.int32(0),
                           DeadCellMonitor_neighbor_maxCellEnergy       = cms.untracked.double(3.),
                           DeadCellMonitor_neighbor_minNeighborEnergy   = cms.untracked.double(1.),
                           DeadCellMonitor_neighbor_minGoodNeighborFrac = cms.untracked.double(.7),
                           DeadCellMonitor_neighbor_maxEnergyFrac       = cms.untracked.double(.2),
                           # HB neighbor flags
                           DeadCellMonitor_HB_neighbor_deltaIeta           = cms.untracked.int32(1),
                           DeadCellMonitor_HB_neighbor_deltaIphi           = cms.untracked.int32(1),
                           DeadCellMonitor_HB_neighbor_deltaDepth          = cms.untracked.int32(0),
                           DeadCellMonitor_HB_neighbor_maxCellEnergy       = cms.untracked.double(3.),
                           DeadCellMonitor_HB_neighbor_minNeighborEnergy   = cms.untracked.double(1.),
                           DeadCellMonitor_HB_neighbor_minGoodNeighborFrac = cms.untracked.double(.7),
                           DeadCellMonitor_HB_neighbor_maxEnergyFrac       = cms.untracked.double(.2),
                           # HE neighbor flags
                           DeadCellMonitor_HE_neighbor_deltaIeta           = cms.untracked.int32(1),
                           DeadCellMonitor_HE_neighbor_deltaIphi           = cms.untracked.int32(1),
                           DeadCellMonitor_HE_neighbor_deltaDepth          = cms.untracked.int32(0),
                           DeadCellMonitor_HE_neighbor_maxCellEnergy       = cms.untracked.double(3.),
                           DeadCellMonitor_HE_neighbor_minNeighborEnergy   = cms.untracked.double(1.),
                           DeadCellMonitor_HE_neighbor_minGoodNeighborFrac = cms.untracked.double(.7),
                           DeadCellMonitor_HE_neighbor_maxEnergyFrac       = cms.untracked.double(.2),
                           # HO neighbor flags
                           DeadCellMonitor_HO_neighbor_deltaIeta           = cms.untracked.int32(1),
                           DeadCellMonitor_HO_neighbor_deltaIphi           = cms.untracked.int32(1),
                           DeadCellMonitor_HO_neighbor_deltaDepth          = cms.untracked.int32(0),
                           DeadCellMonitor_HO_neighbor_maxCellEnergy       = cms.untracked.double(3.),
                           DeadCellMonitor_HO_neighbor_minNeighborEnergy   = cms.untracked.double(1.),
                           DeadCellMonitor_HO_neighbor_minGoodNeighborFrac = cms.untracked.double(.7),
                           DeadCellMonitor_HO_neighbor_maxEnergyFrac       = cms.untracked.double(.2),
                           # HF neighbor flags
                           DeadCellMonitor_HF_neighbor_deltaIeta           = cms.untracked.int32(1),
                           DeadCellMonitor_HF_neighbor_deltaIphi           = cms.untracked.int32(1),
                           DeadCellMonitor_HF_neighbor_deltaDepth          = cms.untracked.int32(1),
                           DeadCellMonitor_HF_neighbor_maxCellEnergy       = cms.untracked.double(3.),
                           DeadCellMonitor_HF_neighbor_minNeighborEnergy   = cms.untracked.double(1.),
                           DeadCellMonitor_HF_neighbor_minGoodNeighborFrac = cms.untracked.double(.7),
                           DeadCellMonitor_HF_neighbor_maxEnergyFrac       = cms.untracked.double(.2),
                           DeadCellMonitor_minErrorFlag                 = cms.untracked.double(0.02),
                           
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

                           HO_NADA_Ecell_frac = cms.untracked.double(0.0),
                           HE_NADA_Ecell_frac = cms.untracked.double(0.0),

                           makeSubdetHistos= cms.untracked.bool(True),
                           HB_NADA_Ecube_frac = cms.untracked.double(0.333),
                           HB_NADA_Ecand_cut2 = cms.untracked.double(10.0),
                           HB_NADA_Ecand_cut0 = cms.untracked.double(1.5),
                           HB_NADA_Ecand_cut1 = cms.untracked.double(2.25),
                           HF_NADA_Ecube_cut = cms.untracked.double(0.1),
                           HOthresholds = cms.untracked.vdouble(2.0, 3.0, 1.0, 5.0, 10.0),
                           ped_Nsigma = cms.untracked.double(-3.1),
                           HotCells = cms.untracked.vstring(),
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

    # Insidious python-ness:  You need to make a copy of the process.minErrorFlag, etc. variables,
    # or future changes to PedestalMonitor_minErrorFlag will also change minErrorFlag!

    # set minimum error value
    minErrorFlag = deepcopy(process.minErrorFlag)
    process.PedestalMonitor_minErrorFlag = minErrorFlag
    process.DeadCellMonitor_minErrorFlag = minErrorFlag

    # set checkNevents
    checkNevents = deepcopy(process.checkNevents)
    process.PedestalMonitor_checkNevents = checkNevents
    process.DeadCellMonitor_checkNevents = checkNevents
    process.DeadCellMonitor_checkNevents_occupancy = checkNevents
    process.DeadCellMonitor_checkNevents_pedestal  = checkNevents
    process.DeadCellMonitor_checkNevents_neighbor  = checkNevents
    process.DeadCellMonitor_checkNevents_energy    = checkNevents

    # set pedestalsInFC
    pedestalsInFC = deepcopy(process.pedestalsInFC)
    process.PedestalMonitor_pedestalsInFC=pedestalsInFC
    process.DeadCellMonitor_pedestalsInFC=pedestalsInFC

    # set makeDiagnoticPlots
    makeDiagnosticPlots = deepcopy(process.MakeDiagnosticPlots)
    process.DeadCellMonitor_makeDiagnosticPlots = makeDiagnosticPlots

    # Set HB/HE/HO/HF
    nsigma = deepcopy(process.DeadCellMonitor_pedestal_Nsigma)
    process.DeadCellMonitor_pedestal_HB_Nsigma           = nsigma
    process.DeadCellMonitor_pedestal_HE_Nsigma           = nsigma
    process.DeadCellMonitor_pedestal_HO_Nsigma           = nsigma
    process.DeadCellMonitor_pedestal_HF_Nsigma           = nsigma
                                                                                      
    return 
