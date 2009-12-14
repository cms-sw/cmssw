import FWCore.ParameterSet.Config as cms
from copy import deepcopy

hcalMonitor = cms.EDFilter("HcalMonitorModule",

                           # GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different values can trigger different levels of messaging

                           Online = cms.untracked.bool(False), # control online/offline differences in code
                           AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           # Calib Types defined in DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h
                           periodicReset = cms.untracked.int32(-1),
                           
                           # eta runs from -43->+43  (-41 -> +41 for HCAL, plus ZDC, which we put at |eta|=43.
                           # add one empty bin beyond that for histogramming prettiness 
                           MaxEta = cms.untracked.double(44.5),
                           MinEta = cms.untracked.double(-44.5),
                           # likewise, phi runs from 1-72.  Add some buffering bins around that region 
                           MaxPhi = cms.untracked.double(73.5),
                           MinPhi = cms.untracked.double(-0.5),

                           # number of luminosity blocks to check
                           Nlumiblocks = cms.untracked.int32(1000),
                           
                           # Determine whether or not to check individual subdetectors
                           checkHF = cms.untracked.bool(True),
                           checkHE = cms.untracked.bool(True),
                           checkHB = cms.untracked.bool(True),
                           checkHO = cms.untracked.bool(True),
                           checkZDC= cms.untracked.bool(False),
                           MonitorDaemon = cms.untracked.bool(True),
                           HcalAnalysis = cms.untracked.bool(False),
                           BadCells = cms.untracked.vstring(),
                           checkNevents = cms.untracked.int32(1000),
                           subSystemFolder = cms.untracked.string("Hcal"),
                           
                           FEDRawDataCollection = cms.untracked.InputTag("source"),
                           
                           #minimum Error Rate that will cause problem histograms to be filled.  Should normally be 0, or close to it?
                           minErrorFlag = cms.untracked.double(0.05), 

                           # Turn on/off timing diagnostic info
                           diagnosticPrescaleLS = cms.untracked.int32(-1),
                           diagnosticPrescaleEvt = cms.untracked.int32(-1),
                           
                           showTiming          = cms.untracked.bool(False), # shows time taken by each process
                           # Make expert-level diagnostic plots (enabling this may drastically slow code!)
                           makeDiagnosticPlots = cms.untracked.bool(False),
                           
                           # Specify Pedestal Units
                           pedestalsInFC                               = cms.untracked.bool(True),
                           # Specify Digis
                           digiLabel = cms.InputTag("hcalDigis"),
                           #Specify RecHits
                           hbheRecHitLabel = cms.InputTag("hbhereco"),
                           hoRecHitLabel = cms.InputTag("horeco"),
                           hfRecHitLabel = cms.InputTag("hfreco"),
                           zdcRecHitLabel = cms.InputTag("zdcreco"),                           
                           hcalLaserLabel = cms.InputTag("hcalLaserReco"),                       
                           #TRIGGER EMULATOR
                           emulTPLabel = cms.InputTag("valHcalTriggerPrimitiveDigis"),

                           # PEDESTAL MONITOR
                           ReferencePedestalMonitor                              = cms.untracked.bool(True),
                           ReferencePedestalMonitor_pedestalsPerChannel          = cms.untracked.bool(True), # not used
                           ReferencePedestalMonitor_pedestalsInFC                = cms.untracked.bool(True),
                           ReferencePedestalMonitor_nominalPedMeanInADC          = cms.untracked.double(3.),
                           ReferencePedestalMonitor_nominalPedWidthInADC         = cms.untracked.double(1.),
                           ReferencePedestalMonitor_maxPedMeanDiffADC            = cms.untracked.double(1.),
                           ReferencePedestalMonitor_maxPedWidthDiffADC           = cms.untracked.double(1.),
                           ReferencePedestalMonitor_startingTimeSlice            = cms.untracked.int32(0),
                           ReferencePedestalMonitor_endingTimeSlice              = cms.untracked.int32(1),
                           ReferencePedestalMonitor_minErrorFlag                 = cms.untracked.double(0.05),
                           ReferencePedestalMonitor_checkNevents                 = cms.untracked.int32(1000),
                           ReferencePedestalMonitor_minEntriesPerPed             = cms.untracked.uint32(100),
                           ReferencePedestalMonitor_makeDiagnosticPlots          = cms.untracked.bool(False),
                           ReferencePedestalMonitor_AllowedCalibTypes = cms.untracked.vint32(1), # Allowed calibration types (empty vector means all types allowed)
                           
                           # DEAD CELL MONITOR
                           DeadCellMonitor                              = cms.untracked.bool(True),
                           DeadCellMonitor_makeDiagnosticPlots          = cms.untracked.bool(False),
                           DeadCellMonitor_test_neverpresent            = cms.untracked.bool(True),
                           DeadCellMonitor_test_digis                   = cms.untracked.bool(True), # tests for digis missing for an entire lumi block
                           DeadCellMonitor_test_rechits                 = cms.untracked.bool(False), # test for rechits less than some energy for an entire lumi block
                           DeadCellMonitor_checkNevents                 = cms.untracked.int32(1000),
                           DeadCellMonitor_minEvents                     = cms.untracked.int32(500), # minimum number of events that must be present in a LB for recent dead cell checks to be made
                           # Checking for cells consistently below energy threshold
                           DeadCellMonitor_energyThreshold              = cms.untracked.double(-1.),
                           DeadCellMonitor_HB_energyThreshold           = cms.untracked.double(-1.),
                           DeadCellMonitor_HE_energyThreshold           = cms.untracked.double(-1.), 
                           DeadCellMonitor_HO_energyThreshold           = cms.untracked.double(-1.),
                           DeadCellMonitor_HF_energyThreshold           = cms.untracked.double(-1.),

                           DeadCellMonitor_minErrorFlag                    = cms.untracked.double(0.05),
                           DeadCellMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           # HOT CELL MONITOR
                           HotCellMonitor                              = cms.untracked.bool(True),
                           HotCellMonitor_makeDiagnosticPlots          = cms.untracked.bool(False),
                           HotCellMonitor_test_neighbor                = cms.untracked.bool(False),
                           HotCellMonitor_test_energy                  = cms.untracked.bool(True),
                           HotCellMonitor_test_persistent              = cms.untracked.bool(True),
                           HotCellMonitor_checkNevents                 = cms.untracked.int32(1000),
                           
                           # Checking for cells above energy threshold at any time
                           # energies raised due to looser cosmics timing
                           HotCellMonitor_energyThreshold              = cms.untracked.double(10.),
                           HotCellMonitor_HB_energyThreshold           = cms.untracked.double(10.),
                           HotCellMonitor_HE_energyThreshold           = cms.untracked.double(10.), 
                           HotCellMonitor_HO_energyThreshold           = cms.untracked.double(10.),
                           HotCellMonitor_HF_energyThreshold           = cms.untracked.double(20.),
                           HotCellMonitor_ZDC_energyThreshold          = cms.untracked.double(999.), # not yet implemented
                           # Checking for cells consistently babove energy threshold
                           HotCellMonitor_persistentThreshold              = cms.untracked.double(6.),
                           HotCellMonitor_HB_persistentThreshold           = cms.untracked.double(6.),
                           HotCellMonitor_HE_persistentThreshold           = cms.untracked.double(6.), 
                           HotCellMonitor_HO_persistentThreshold           = cms.untracked.double(6.),
                           HotCellMonitor_HF_persistentThreshold           = cms.untracked.double(10.),
                           HotCellMonitor_ZDC_persistentThreshold          = cms.untracked.double(999.), # not yet implemented

                           HotCellMonitor_HO_SiPMscalefactor               = cms.untracked.double(1.), # scale factor to apply to energy threshold for SiPMs (when SiPMs weren't properly calibrated)
                           HotCellMonitor_HFfwdScale                       = cms.untracked.double(1.), # scale factor to raise energy thresholds for HF cells at |ieta| = 40,41
                           
                           # Check for cells above their neighbors -- not currently in use
                           HotCellMonitor_neighbor_deltaIeta           = cms.untracked.int32(1),
                           HotCellMonitor_neighbor_deltaIphi           = cms.untracked.int32(1),
                           HotCellMonitor_neighbor_deltaDepth          = cms.untracked.int32(4),
                           HotCellMonitor_neighbor_minCellEnergy       = cms.untracked.double(0.),
                           HotCellMonitor_neighbor_minNeighborEnergy   = cms.untracked.double(0.),
                           HotCellMonitor_neighbor_maxEnergy           = cms.untracked.double(25),
                           HotCellMonitor_neighbor_HotEnergyFrac       = cms.untracked.double(.02),
                           # HB neighbor flags
                           HotCellMonitor_HB_neighbor_deltaIeta           = cms.untracked.int32(1),
                           HotCellMonitor_HB_neighbor_deltaIphi           = cms.untracked.int32(1),
                           HotCellMonitor_HB_neighbor_deltaDepth          = cms.untracked.int32(4),
                           HotCellMonitor_HB_neighbor_minCellEnergy       = cms.untracked.double(2.),
                           HotCellMonitor_HB_neighbor_minNeighborEnergy   = cms.untracked.double(0.),
                           HotCellMonitor_HB_neighbor_maxEnergy           = cms.untracked.double(25),
                           HotCellMonitor_HB_neighbor_HotEnergyFrac       = cms.untracked.double(.02),
                           # HE neighbor flags
                           HotCellMonitor_HE_neighbor_deltaIeta           = cms.untracked.int32(1),
                           HotCellMonitor_HE_neighbor_deltaIphi           = cms.untracked.int32(1),
                           HotCellMonitor_HE_neighbor_deltaDepth          = cms.untracked.int32(4),
                           HotCellMonitor_HE_neighbor_minCellEnergy       = cms.untracked.double(2.),
                           HotCellMonitor_HE_neighbor_minNeighborEnergy   = cms.untracked.double(0.),
                           HotCellMonitor_HE_neighbor_maxEnergy           = cms.untracked.double(25),
                           HotCellMonitor_HE_neighbor_HotEnergyFrac       = cms.untracked.double(.02),
                           # HO neighbor flags
                           HotCellMonitor_HO_neighbor_deltaIeta           = cms.untracked.int32(1),
                           HotCellMonitor_HO_neighbor_deltaIphi           = cms.untracked.int32(1),
                           HotCellMonitor_HO_neighbor_deltaDepth          = cms.untracked.int32(4),
                           HotCellMonitor_HO_neighbor_minCellEnergy       = cms.untracked.double(5.),
                           HotCellMonitor_HO_neighbor_minNeighborEnergy   = cms.untracked.double(0.),
                           HotCellMonitor_HO_neighbor_maxEnergy           = cms.untracked.double(25),
                           HotCellMonitor_HO_neighbor_HotEnergyFrac       = cms.untracked.double(.02),
                           # HF neighbor flags
                           HotCellMonitor_HF_neighbor_deltaIeta           = cms.untracked.int32(1),
                           HotCellMonitor_HF_neighbor_deltaIphi           = cms.untracked.int32(1),
                           HotCellMonitor_HF_neighbor_deltaDepth          = cms.untracked.int32(4),
                           HotCellMonitor_HF_neighbor_minCellEnergy       = cms.untracked.double(2.),
                           HotCellMonitor_HF_neighbor_minNeighborEnergy   = cms.untracked.double(0.),
                           HotCellMonitor_HF_neighbor_maxEnergy           = cms.untracked.double(25),
                           HotCellMonitor_HF_neighbor_HotEnergyFrac       = cms.untracked.double(.02),
                           
                           HotCellMonitor_minErrorFlag                    = cms.untracked.double(0.05),
                           HotCellMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)

                           # DIGI MONITOR
                           DigiMonitor                                    = cms.untracked.bool(True),
                           DigiMonitor_checkNevents                       = cms.untracked.int32(1000),
                           DigiMonitor_problems_checkForMissingDigis      = cms.untracked.bool(False), # also checked in DeadCellMonitor, which may be the more appropriate place for the check
                           DigiMonitor_problems_checkCapID                = cms.untracked.bool(True),
                           DigiMonitor_problems_checkDigiSize             = cms.untracked.bool(True),
                           DigiMonitor_problems_checkADCsum               = cms.untracked.bool(True),
                           DigiMonitor_problems_checkDVerr                = cms.untracked.bool(True),
                           DigiMonitor_minDigiSize                        = cms.untracked.int32(10),
                           DigiMonitor_maxDigiSize                        = cms.untracked.int32(10),
                           # ADC counts must be above the threshold values below for appropriate histograms to be filled
                           DigiMonitor_shapeThresh                        = cms.untracked.int32(50),
                           DigiMonitor_shapeThreshHB                      = cms.untracked.int32(50),
                           DigiMonitor_shapeThreshHE                      = cms.untracked.int32(50),
                           DigiMonitor_shapeThreshHO                      = cms.untracked.int32(50),
                           DigiMonitor_shapeThreshHF                      = cms.untracked.int32(50),
                           DigiMonitor_makeDiagnosticPlots                 = cms.untracked.bool(False), 
                           DigiMonitor_DigisPerChannel                    = cms.untracked.bool(False), # not currently used
                           DigiMonitor_ExpectedOrbitMessageTime           = cms.untracked.int32(-1),
                           DigiMonitor_shutOffOrbitTest                   = cms.untracked.bool(False),
                           DigiMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           # RECHIT MONITOR
                           RecHitMonitor                                  = cms.untracked.bool(True),
                           RecHitMonitor_checkNevents                     = cms.untracked.int32(1000),
                           RecHitMonitor_minErrorFlag                     = cms.untracked.double(0.),
                           RecHitMonitor_makeDiagnosticPlots              = cms.untracked.bool(False),
                           RecHitMonitor_energyThreshold                  = cms.untracked.double(2.),
                           RecHitMonitor_HB_energyThreshold               = cms.untracked.double(2.),
                           RecHitMonitor_HO_energyThreshold               = cms.untracked.double(2.),
                           RecHitMonitor_HE_energyThreshold               = cms.untracked.double(2.),
                           RecHitMonitor_HF_energyThreshold               = cms.untracked.double(2.),
                           RecHitMonitor_ZDC_energyThreshold              = cms.untracked.double(2.),
                           RecHitMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           # BEAM MONITOR
                           BeamMonitor                                    = cms.untracked.bool(True),
                           BeamMonitor_checkNevents                       = cms.untracked.int32(1000),
                           BeamMonitor_minErrorFlag                       = cms.untracked.double(0.20),
                           BeamMonitor_makeDiagnosticPlots                = cms.untracked.bool(False),
                           BeamMonitor_lumiprescale                       = cms.untracked.int32(1), # set number of bins in Lumi-section plots -- divide Nlumiblocks by this prescale
                           BeamMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)

                           BeamMonitor_lumiqualitydir = cms.untracked.string(""),
                           BeamMonitor_minEvents = cms.untracked.int32(500),
                           # DATA FORMAT MONITOR
                           DataFormatMonitor                              = cms.untracked.bool(True),
                           DataFormatMonitor_checkNevents                 = cms.untracked.int32(1000),
                           dfPrtLvl                                       = cms.untracked.int32(0), # this seems similar to the debug int we have; deprecate this?
                           DataFormatMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           # ZDC MONITOR
                           ZDCMonitor                                     = cms.untracked.bool(False),
                           ZDCMonitor_checkNevents                        = cms.untracked.int32(1000),
                           ZDCMonitor_deadthresholdrate                   = cms.untracked.double(0.),
                                                      

                           # DATA INTEGRITY TASK
                           DataIntegrityTask = cms.untracked.bool(False),

                           # TRIG PRIM MONITOR
                           TrigPrimMonitor = cms.untracked.bool(False),
                           TrigPrimMonitor_OccThresh                      = cms.untracked.double(1.),
                           TrigPrimMonitor_Threshold                      = cms.untracked.double(1.),
                           TrigPrimMonitor_TPdigiTS                       = cms.untracked.int32(1),
                           TrigPrimMonitor_ADCdigiTS                      = cms.untracked.int32(3),
                           TrigPrimMonitor_checkNevents                   = cms.untracked.int32(1000),
                           TrigPrimMonitor_makeDiagnosticPlots            = cms.untracked.bool(False),
                           TrigPrimMonitor_ZSAlarmThreshold               = cms.untracked.int32(12),
                           TrigPrimMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)
                           gtLabel = cms.InputTag("l1GtUnpack"),
                           
                           # LED MONITOR
                           LEDMonitor = cms.untracked.bool(False),
                           LED_ADC_Thresh = cms.untracked.double(-1000.0),
                           LEDPerChannel = cms.untracked.bool(True),
                           LEDMonitor_AllowedCalibTypes = cms.untracked.vint32(), # Allowed calibration types (empty vector means all types allowed)  # Not yet in use

                           #LASER MONITOR
                           LaserMonitor = cms.untracked.bool(False),
                           Laser_ADC_Thresh = cms.untracked.double(-1000.0),
                           LaserPerChannel = cms.untracked.bool(True),
                           LaserMonitor_AllowedCalibTypes = cms.untracked.vint32(2,3,4,5), # Allowed calibration types (empty vector means all types allowed)
                           # SPECIALIZED (EXPERT-USE) MONITORS

                            # EXPERT MONITOR (should generally be turned off)
                           ExpertMonitor = cms.untracked.bool(False),

                           # Empty Event/Unsuppressed data monitor
                           EEUSMonitor = cms.untracked.bool(False),
			   
			   # NZS Monitor
		           NZSMonitor = cms.untracked.bool(False),
			   hltResultsTag=cms.untracked.InputTag("TriggerResults","","HLT"),
			   NZSMonitor_nzsHLTnames = cms.untracked.vstring('HLT_HcalPhiSym','HLT_HcalNZS'),
			   NZSMonitor_NZSeventPeriod = cms.untracked.int32(4096),

			   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
                           # Detector diagnostic Monitors  
                           DetDiagPedestalMonitor = cms.untracked.bool(False),
                           DetDiagLEDMonitor      = cms.untracked.bool(False),
                           DetDiagLaserMonitor    = cms.untracked.bool(False),
			   DetDiagNoiseMonitor    = cms.untracked.bool(False),
			   DetDiagTimingMonitor   = cms.untracked.bool(False),
			   
                           UseDB                  = cms.untracked.bool(False),
                           LEDReferenceData       = cms.untracked.string("./"),
                           PedReferenceData       = cms.untracked.string("./"),
			   LaserReferenceData     = cms.untracked.string("./"),
                           OutputFilePath         = cms.untracked.string("./"),
                           
                           LEDDeltaTreshold       = cms.untracked.double(7.0),
                           LEDRmsTreshold         = cms.untracked.double(5.0),

                           HBMeanPedestalTreshold = cms.untracked.double(0.1),
                           HBRmsPedestalTreshold  = cms.untracked.double(0.1),
                           HEMeanPedestalTreshold = cms.untracked.double(0.1),
                           HERmsPedestalTreshold  = cms.untracked.double(0.1),
                           HOMeanPedestalTreshold = cms.untracked.double(0.1),
                           HORmsPedestalTreshold  = cms.untracked.double(0.1),
                           HFMeanPedestalTreshold = cms.untracked.double(0.1),
                           HFRmsPedestalTreshold  = cms.untracked.double(0.1),
                           # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
			   
                           # ------------- DEPRECATED/UNUSED MONITORS ----------------------- #
                                                      
                           # CALO TOWER MONITOR
                           CaloTowerMonitor = cms.untracked.bool(False),
                           caloTowerLabel = cms.InputTag("towerMaker"),

                           # MTCC MONITOR
                           MTCCMonitor = cms.untracked.bool(False),
                           MTCCOccThresh = cms.untracked.double(10.0),
                           DumpPhiLow = cms.untracked.int32(10),
                           DumpPhiHigh = cms.untracked.int32(10),
                           DumpEtaLow = cms.untracked.int32(0),
                           DumpEtaHigh = cms.untracked.int32(10),
                           DumpThreshold = cms.untracked.double(500.0),
                           # --------------------------------------------------------------- #

                           HLTriggerResults                    = cms.untracked.InputTag("TriggerResults","","HLT"),
                           MetSource                           = cms.untracked.InputTag("met"),
                           JetSource                           = cms.untracked.InputTag("iterativeCone5CaloJets"),
                           TrackSource                         = cms.untracked.InputTag("generalTracks"),
                           rbxCollName                         = cms.untracked.string('hcalnoise'),
                           TriggerRequirement                  = cms.untracked.string("HLT_MET100"),
		           UseMetCutInsteadOfTrigger	       = cms.untracked.bool(True),
		           MetCut			       = cms.untracked.double(0.0),
                           JetMinEt                            = cms.untracked.double(20.0),
                           JetMaxEta                           = cms.untracked.double(2.0),
                           ConstituentsToJetMatchingDeltaR     = cms.untracked.double(0.5),
                           TrackMaxIp                          = cms.untracked.double(0.1),
                           TrackMinThreshold                   = cms.untracked.double(1.0),
                           MinJetChargeFraction                = cms.untracked.double(0.05),
                           MaxJetHadronicEnergyFraction        = cms.untracked.double(0.98),
                           caloTowerCollName		       = cms.InputTag("towerMaker"),

                           )


def setHcalTaskValues(process):
    # If you import this function directly, you can then set all the individual subtask values to the global settings
    # (This is useful if you've changed the global value, and you want it to propagate everywhere)

    # Set minimum value needed to put an entry into Problem histograms.  (values are between 0-1)

    # Insidious python-ness:  You need to make a copy of the process.minErrorFlag, etc. variables,
    # or future changes to PedestalMonitor_minErrorFlag will also change minErrorFlag!

    # set minimum error value -- we may want to keep them assigned explicitly as above
    #minErrorFlag = deepcopy(process.minErrorFlag.value())
    #process.BeamMonitor_minErrorFlag     = cms.untracked.double(minErrorFlag)
    #process.DeadCellMonitor_minErrorFlag = cms.untracked.double(minErrorFlag)
    #process.HotCellMonitor_minErrorFlag  = cms.untracked.double(minErrorFlag)
    #process.ReferencePedestalMonitor_minErrorFlag = cms.untracked.double(minErrorFlag)
    #process.RecHitMonitor_minErrorFlag   = cms.untracked.double(minErrorFlag)

    # set checkNevents -- soon to be deprecated in favor of checking once/lum'y block
    checkNevents = deepcopy(process.checkNevents.value())
    process.BeamMonitor_checkNevents                      = checkNevents
    process.DataFormatMonitor_checkNevents                = checkNevents
    process.DeadCellMonitor_checkNevents                  = checkNevents
    process.DigiMonitor_checkNevents                      = checkNevents
    process.HotCellMonitor_checkNevents                   = checkNevents
    process.ReferencePedestalMonitor_checkNevents                  = checkNevents
    process.RecHitMonitor_checkNevents                    = checkNevents
    process.TrigPrimMonitor_checkNevents                  = checkNevents
    process.ZDCMonitor_checkNevents                       = checkNevents
    
    # set pedestalsInFC
    pedestalsInFC = deepcopy(process.pedestalsInFC.value())
    process.HotCellMonitor_pedestalsInFC  = cms.untracked.bool(pedestalsInFC)
    process.ReferencePedestalMonitor_pedestalsInFC = cms.untracked.bool(pedestalsInFC)

    # set makeDiagnoticPlots
    makeDiagnosticPlots                         = deepcopy(process.makeDiagnosticPlots.value())

    process.BeamMonitor_makeDiagnosticPlots     = makeDiagnosticPlots
    process.DeadCellMonitor_makeDiagnosticPlots = makeDiagnosticPlots
    process.DigiMonitor_makeDiagnosticPlots     = makeDiagnosticPlots
    process.HotCellMonitor_makeDiagnosticPlots  = makeDiagnosticPlots
    process.RecHitMonitor_makeDiagnosticPlots   = makeDiagnosticPlots
    process.TrigPrimMonitor_makeDiagnosticPlots = makeDiagnosticPlots
    process.ReferencePedestalMonitor_makeDiagnosticPlots = makeDiagnosticPlots
    
    return


def setHcalSubdetTaskValues(process):
    # Set HB/HE/HO/HF

    # Dead Cell Monitor
    dead_energyThreshold = deepcopy(process.DeadCellMonitor_energyThreshold.value())
    process.DeadCellMonitor_HB_energyThreshold           = cms.untracked.double(dead_energyThreshold)
    process.DeadCellMonitor_HE_energyThreshold           = cms.untracked.double(dead_energyThreshold)
    process.DeadCellMonitor_HO_energyThreshold           = cms.untracked.double(dead_energyThreshold)
    process.DeadCellMonitor_HF_energyThreshold           = cms.untracked.double(dead_energyThreshold)
    process.DeadCellMonitor_ZDC_energyThreshold          = cms.untracked.double(dead_energyThreshold)

    # Hot Cell Monitor
    hot_energyThreshold = deepcopy(process.HotCellMonitor_energyThreshold.value())
    process.HotCellMonitor_HB_energyThreshold           = cms.untracked.double(hot_energyThreshold)
    process.HotCellMonitor_HE_energyThreshold           = cms.untracked.double(hot_energyThreshold)
    process.HotCellMonitor_HO_energyThreshold           = cms.untracked.double(hot_energyThreshold)
    process.HotCellMonitor_HF_energyThreshold           = cms.untracked.double(hot_energyThreshold)
    process.HotCellMonitor_ZDC_energyThreshold          = cms.untracked.double(hot_energyThreshold)

    hot_persistentThreshold = deepcopy(process.HotCellMonitor_persistentThreshold.value())
    process.HotCellMonitor_HB_persistentThreshold           = cms.untracked.double(hot_persistentThreshold)
    process.HotCellMonitor_HE_persistentThreshold           = cms.untracked.double(hot_persistentThreshold)
    process.HotCellMonitor_HO_persistentThreshold           = cms.untracked.double(hot_persistentThreshold)
    process.HotCellMonitor_HF_persistentThreshold           = cms.untracked.double(hot_persistentThreshold)
    process.HotCellMonitor_ZDC_persistentThreshold          = cms.untracked.double(hot_persistentThreshold)

    # Rec Hit Monitor
    rechit_energyThreshold = deepcopy(process.RecHitMonitor_energyThreshold.value())
    process.RecHitMonitor_HB_energyThreshold           = cms.untracked.double(rechit_energyThreshold)
    process.RecHitMonitor_HE_energyThreshold           = cms.untracked.double(rechit_energyThreshold)
    process.RecHitMonitor_HO_energyThreshold           = cms.untracked.double(rechit_energyThreshold)
    process.RecHitMonitor_HF_energyThreshold           = cms.untracked.double(rechit_energyThreshold)
    process.RecHitMonitor_ZDC_energyThreshold          = cms.untracked.double(rechit_energyThreshold)
    return 
