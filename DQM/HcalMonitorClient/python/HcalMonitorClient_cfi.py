import FWCore.ParameterSet.Config as cms
from copy import deepcopy

hcalClient = cms.EDFilter("HcalMonitorClient",

                          # Variables for the Overall Client
                          runningStandalone         = cms.untracked.bool(False),
                          processName               = cms.untracked.string(''),
                          inputfile                 = cms.untracked.string(''),
                          baseHtmlDir               = cms.untracked.string('.'),
                          MonitorDaemon             = cms.untracked.bool(True),
                          diagnosticPrescaleTime    = cms.untracked.int32(-1),
                          diagnosticPrescaleEvt     = cms.untracked.int32(200),
                          diagnosticPrescaleLS      = cms.untracked.int32(-1),
                          diagnosticPrescaleUpdate  = cms.untracked.int32(-1),
                          resetFreqTime             = cms.untracked.int32(-1),
                          resetFreqEvents           = cms.untracked.int32(-1),
                          resetFreqLS               = cms.untracked.int32(-1),
                          resetFreqUpdates          = cms.untracked.int32(-1),
                          enableExit                = cms.untracked.bool(False),
                          #DoPerChanTests            = cms.untracked.bool(False), # is this used anywhere?
                          # Variables from which subtasks may inherit
                          subDetsOn                 = cms.untracked.vstring('HB', 'HE', 'HF', 'HO'),
                          debug                     = cms.untracked.int32(0),
                          showTiming                = cms.untracked.bool(False),
                          fillUnphysicalIphi        = cms.untracked.bool(True),
                          BadCells = cms.untracked.vstring(),

                          # Pedestal Client,
                          PedestalClient                       = cms.untracked.bool(True),
                          PedestalClient_nominalPedMeanInADC   = cms.untracked.double(3.),
                          PedestalClient_nominalPedWidthInADC  = cms.untracked.double(1.),
                          PedestalClient_maxPedMeanDiffADC     = cms.untracked.double(1.),
                          PedestalClient_maxPedWidthDiffADC    = cms.untracked.double(1.),
                          PedestalClient_pedestalsInFC         = cms.untracked.bool(True),
                          PedestalClient_startingTimeSlice     = cms.untracked.int32(0),
                          PedestalClient_endingTimeSlice       = cms.untracked.int32(1),
                          PedestalClient_minErrorFlag          = cms.untracked.double(0.05),
                          
                          # DigiClient
                          DigiClient                = cms.untracked.bool(True),
                          #digiErrorFrac             = cms.untracked.double(0.05),
                          #CapIdMEAN_ErrThresh       = cms.untracked.double(1.5),
                          #CapIdRMS_ErrThresh        = cms.untracked.double(0.25),

                          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
                          # Detector diagnostic Monitors  
                          DetDiagPedestalClient     = cms.untracked.bool(False),
                          DetDiagLEDClient          = cms.untracked.bool(False),
                          DetDiagLaserClient        = cms.untracked.bool(False),
                          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

                          # Dead Cell Client
                          DeadCellClient                                = cms.untracked.bool(True),
                          DeadCellClient_test_neverpresent              = cms.untracked.bool(True),
                          DeadCellClient_test_occupancy                 = cms.untracked.bool(True),
                          DeadCellClient_test_energy                    = cms.untracked.bool(True),
                          DeadCellClient_checkNevents                   = cms.untracked.int32(100),
                          DeadCellClient_minErrorFlag                   = cms.untracked.double(0.05),
                          DeadCellClient_makeDiagnosticPlots            = cms.untracked.bool(False),

                          # Hot Cell Client
                          HotCellClient                                 = cms.untracked.bool(True),
                          HotCellClient_test_persistent                 = cms.untracked.bool(True),
                          HotCellClient_test_pedestal                   = cms.untracked.bool(True),
                          HotCellClient_test_energy                     = cms.untracked.bool(True),
                          HotCellClient_test_neighbor                   = cms.untracked.bool(False),
                          HotCellClient_checkNevents                    = cms.untracked.int32(100),
                          HotCellClient_minErrorFlag                    = cms.untracked.double(0.05),
                          HotCellClient_makeDiagnosticPlots             = cms.untracked.bool(False),
                          
                          # DataFormatClient
                          DataFormatClient          = cms.untracked.bool(True),

                          # Summary Client
                          SummaryClient             = cms.untracked.bool(True),

                          #LED Client
                          LEDClient                 = cms.untracked.bool(True),
                          LEDRMS_ErrThresh          = cms.untracked.double(0.8),
                          LEDMEAN_ErrThresh         = cms.untracked.double(2.25),

                          # RecHit Client
                          RecHitClient                                  = cms.untracked.bool(True),
                          RecHitClient_checkNevents                     = cms.untracked.int32(500),
                          RecHitClient_minErrorFlag                     = cms.untracked.double(0.00),
                          RecHitClient_makeDiagnosticPlots              = cms.untracked.bool(False),
                          
                          # CaloTowerClient
                          CaloTowerClient           = cms.untracked.bool(False),

                          # TrigPrimClient
                          TrigPrimClient            = cms.untracked.bool(True),

                          # BeamClient
                          BeamClient                     = cms.untracked.bool(True),
                          BeamClient_checkNevents        = cms.untracked.int32(100),
                          BeamClient_minErrorFlag        = cms.untracked.double(0.05),
                          BeamClient_makeDiagosticPlots  = cms.untracked.bool(False)
                          )



def setHcalClientValuesFromMonitor(client, origmonitor, debug=False):
    # need to make separate copy, or changing client after this call will also change monitor!
    monitor=deepcopy(origmonitor)
    
    #Reads variables from monitor module, and sets the client's copy of those variables to the same value.
    #This way, when you disable the DataFormat Monitor, the DataFormat client is also turned off automatically, etc.

    # Set update period of client to checkNevents value of monitor 

    # This doesn't work, because monitor.checkNevents returns 'cms.untracked.bool(...)'
    #client.diagnosticPrescaleEvt                  = max(100,monitor.checkNevents) # combine checkNevents and diagnosticPrescaleEvt into one?
    checkN = deepcopy(client.diagnosticPrescaleEvt)
    # Beam, RecHit checkNevents not used; we could get rid of them
    client.BeamClient_checkNevents                = checkN
    client.RecHitClient_checkNevents              = checkN
    # Hot and Dead cell values only used for labelling
    client.DeadCellClient_checkNevents            = monitor.DeadCellMonitor_checkNevents
    client.HotCellClient_checkNevents            = monitor.HotCellMonitor_checkNevents
    

    client.fillUnphysicalIphi                     = monitor.fillUnphysicalIphi 

    

    # Beam Client
    client.BeamClient                             = monitor.BeamMonitor
    client.BeamClient_minErrorFlag                = monitor.BeamMonitor_minErrorFlag
    client.BeamClient_makeDiagnosticPlots         = monitor.BeamMonitor_makeDiagnosticPlots
    
    # Dead Cell
    client.DeadCellClient                         = monitor.DeadCellMonitor
    client.DeadCellClient_test_neverpresent       = monitor.DeadCellMonitor_test_neverpresent
    client.DeadCellClient_test_occupancy          = monitor.DeadCellMonitor_test_occupancy
    client.DeadCellClient_test_energy             = monitor.DeadCellMonitor_test_energy
    #client.DeadCellClient_minErrorFlag           = monitor.DeadCellMonitor_minErrorFlag # want to keep these separate?
    client.DeadCellClient_makeDiagnosticPlots     = monitor.DeadCellMonitor_makeDiagnosticPlots          

    # Digi 
    client.DigiClient                             = monitor.DigiMonitor

    # Hot Cell
    client.HotCellClient                          = monitor.HotCellMonitor
    client.HotCellClient_test_persistent          = monitor.HotCellMonitor_test_persistent
    client.HotCellClient_test_pedestal            = monitor.HotCellMonitor_test_pedestal
    client.HotCellClient_test_energy              = monitor.HotCellMonitor_test_energy
    client.HotCellClient_test_neighbor            = monitor.HotCellMonitor_test_neighbor
    #client.HotCellClient_minErrorFlag            = monitor.HotCellMonitor_minErrorFlag # want to keep these separate?
    client.HotCellClient_makeDiagnosticPlots      = monitor.HotCellMonitor_makeDiagnosticPlots

    # Pedestal Client
    client.PedestalClient                         = monitor.PedestalMonitor
    client.PedestalClient_nominalPedMeanInADC     = monitor.PedestalMonitor_nominalPedMeanInADC
    client.PedestalClient_nominalPedWidthInADC    = monitor.PedestalMonitor_nominalPedWidthInADC
    client.PedestalClient_maxPedMeanDiffADC       = monitor.PedestalMonitor_maxPedMeanDiffADC
    client.PedestalClient_maxPedWidthDiffADC      = monitor.PedestalMonitor_maxPedWidthDiffADC
    client.PedestalClient_pedestalsInFC           = monitor.PedestalMonitor_pedestalsInFC
    client.PedestalClient_startingTimeSlice       = monitor.PedestalMonitor_startingTimeSlice
    client.PedestalClient_endingTimeSlice         = monitor.PedestalMonitor_endingTimeSlice
    #client.PedestalClient_minErrorFlag           = monitor.PedestalMonitor_minErrorFlag # want to keep these separate?

    # Rec Hit Client
    client.RecHitClient                           = monitor.RecHitMonitor
    client.RecHitClient_minErrorFlag              = monitor.RecHitMonitor_minErrorFlag
    client.RecHitClient_makeDiagnosticPlots       = monitor.RecHitMonitor_makeDiagnosticPlots


    client.DataFormatClient  = monitor.DataFormatMonitor
    client.LEDClient         = monitor.LEDMonitor
    client.CaloTowerClient   = monitor.CaloTowerMonitor
    client.TrigPrimClient    = monitor.TrigPrimMonitor

    client.showTiming        = monitor.showTiming
    client.debug             = monitor.debug

    if (debug):
        print "HcalMonitorClient values from HcalMonitorModule: "
        print "Debug              = ", client.debug
        print "showTiming         = ", client.showTiming
        print "PrescaleEvt        = ", client.diagnosticPrescaleEvt
        print "Pedestal Client    = ", client.PedestalClient
        print "Digi Client        = ", client.DigiClient
        print "DeadCell Client    = ", client.DeadCellClient
        print "\t\t Test DeadCell occupancy? ", client.DeadCellClient_test_occupancy
        print "\t\t Test DeadCell pedestal? ", client.DeadCellClient_test_pedestal
        print "\t\t Test DeadCell energy? ", client.DeadCellClient_test_energy
        print "\t\t Test DeadCell neighbor? ", client.DeadCellClient_test_neighbor
        print "\t\t Min Error Flag  = ",client.DeadCellClient_minErrorFlag
        print "\t\t make diagnostics? ",client.DeadCellClient_makeDiagnosticPlots

        print "HotCell Client    = ", client.HotCellClient
        print "\t\t Test HotCell persistently above threshold? ", client.HotCellClient_test_persistent
        print "\t\t Test HotCell pedestal? ",                     client.HotCellClient_test_pedestal
        print "\t\t Test HotCell energy? ",                       client.HotCellClient_test_energy
        print "\t\t Test HotCell neighbor? ",                     client.HotCellClient_test_neighbor
        print "\t\t Min Error Flag  = ",                          client.HotCellClient_minErrorFlag
        print "\t\t make diagnostics? ",                          client.HotCellClient_makeDiagnosticPlots
                                                                                        
        print "DataFormat Client  = ",   client.DataFormatClient
        print "Summary Client     = ",   client.SummaryClient
        print "LED Client         = ",   client.LEDClient
        print "RecHit Client      = ",   client.RecHitClient
        print "\t\t CheckNevents  = ",   client.RecHitClient_checkNevents
        print "\t\t MinErrorFlag  = ",   client.RecHitClient_minErrorFlag
        print "\t\t make diagnostics? ", client.RecHitClient_makeDiagnosticPlots
        print "Beam Client      = ",     client.BeamClient
        print "\t\t CheckNevents  = ",   client.BeamClient_checkNevents
        print "\t\t MinErrorFlag  = ",   client.BeamClient_minErrorFlag
        print "\t\t make diagnostics? ", client.BeamClient_makeDiagnosticPlots
        print "CaloTower Client   = ",   client.CaloTowerClient
        print "TrigPrim Client    = ",   client.TrigPrimClient

    return
