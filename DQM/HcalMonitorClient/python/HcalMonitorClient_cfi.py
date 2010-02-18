import FWCore.ParameterSet.Config as cms

from copy import deepcopy

hcalClient = cms.EDAnalyzer("HcalMonitorClient",

                          # Variables for the Overall Client
                          runningStandalone         = cms.untracked.bool(False),
                          Online                    = cms.untracked.bool(False), 
                          databasedir               = cms.untracked.string(''),

                          # maximum number of lumi blocks to appear in some histograms
                          Nlumiblocks = cms.untracked.int32(1000),
                          subSystemFolder           = cms.untracked.string('Hcal'),
                          processName               = cms.untracked.string(''),
                          inputfile                 = cms.untracked.string(''),
                          baseHtmlDir               = cms.untracked.string('.'),
                          MonitorDaemon             = cms.untracked.bool(True),

                          # run actual client either every N events or M lumi blocks (or both)
                          diagnosticPrescaleEvt     = cms.untracked.int32(-1),
                          diagnosticPrescaleLS      = cms.untracked.int32(1),
                          resetFreqEvents           = cms.untracked.int32(-1),
                          resetFreqLS               = cms.untracked.int32(-1),
                          
                          # Variables from which subtasks may inherit
                          subDetsOn                 = cms.untracked.vstring('HB', 'HE', 'HF', 'HO'), # we should get rid of this at some point
                          debug                     = cms.untracked.int32(0),
                          showTiming                = cms.untracked.bool(False),

                          BadCells = cms.untracked.vstring(),

                          # Reference Pedestal Client,
                          ReferencePedestalClient                       = cms.untracked.bool(True),
                          ReferencePedestalClient_nominalPedMeanInADC   = cms.untracked.double(3.),
                          ReferencePedestalClient_nominalPedWidthInADC  = cms.untracked.double(1.),
                          ReferencePedestalClient_maxPedMeanDiffADC     = cms.untracked.double(1.),
                          ReferencePedestalClient_maxPedWidthDiffADC    = cms.untracked.double(1.),
                          ReferencePedestalClient_minErrorFlag          = cms.untracked.double(0.05),
                          ReferencePedestalClient_makeDiagnosticPlots   = cms.untracked.bool(False),
                          
                          # DigiClient
                          DigiClient                 = cms.untracked.bool(True),
                          DigiClient_minErrorFlag    = cms.untracked.double(0.05),
 
                          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
                          # Detector diagnostic Monitors  
                          DetDiagPedestalClient     = cms.untracked.bool(False),
                          DetDiagLEDClient          = cms.untracked.bool(False),
                          DetDiagLaserClient        = cms.untracked.bool(False),
                          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

                          # Dead Cell Client
                          DeadCellClient                                = cms.untracked.bool(True),
                          DeadCellClient_test_digis                     = cms.untracked.bool(True),
                          DeadCellClient_test_rechits                   = cms.untracked.bool(True),
                          DeadCellClient_checkNevents                   = cms.untracked.int32(1000),
                          DeadCellClient_minErrorFlag                   = cms.untracked.double(0.05),
                          DeadCellClient_makeDiagnosticPlots            = cms.untracked.bool(False),

                          # Hot Cell Client
                          HotCellClient                                 = cms.untracked.bool(True),
                          HotCellClient_test_persistent                 = cms.untracked.bool(True),
                          #HotCellClient_test_pedestal                   = cms.untracked.bool(True),
                          HotCellClient_test_energy                     = cms.untracked.bool(True),
                          HotCellClient_test_neighbor                   = cms.untracked.bool(False),
                          HotCellClient_checkNevents                    = cms.untracked.int32(1000),
                          HotCellClient_minErrorFlag                    = cms.untracked.double(0.05),
                          HotCellClient_makeDiagnosticPlots             = cms.untracked.bool(False),
                          
                          # DataFormatClient
                          DataFormatClient          = cms.untracked.bool(True),
                          DataFormatClient_minErrorFlag = cms.untracked.double(0.01),
                          
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

                          ########################################################################
                          # Noise Client
                          NoiseClient                                  = cms.untracked.bool(False),
                          ########################################################################
                          
                          # CaloTowerClient
                          CaloTowerClient           = cms.untracked.bool(False),

                          # TrigPrimClient
                          TrigPrimClient            = cms.untracked.bool(True),

                          # BeamClient
                          BeamClient                     = cms.untracked.bool(True),
                          BeamClient_checkNevents        = cms.untracked.int32(100),
                          BeamClient_minErrorFlag        = cms.untracked.double(0.05),
                          BeamClient_makeDiagnosticPlots = cms.untracked.bool(False)
                          )



def setHcalClientValuesFromMonitor(client, origmonitor, debug=False):

    # need to make separate copy, or changing client after this call will also change monitor!
    monitor=deepcopy(origmonitor)
    
    #Reads variables from monitor module, and sets the client's copy of those variables to the same value.
    #This way, when you disable the DataFormat Monitor, the DataFormat client is also turned off automatically, etc.

    client.subSystemFolder = monitor.subSystemFolder

    # Set update period of client to checkNevents value of monitor ?
    checkN = deepcopy(client.diagnosticPrescaleEvt.value())
    
    client.Online                                 = monitor.Online.value()
    client.Nlumiblocks                            = monitor.Nlumiblocks.value()

    # Beam Client
    client.BeamClient                             = monitor.BeamMonitor.value()
    client.BeamClient_minErrorFlag                = monitor.BeamMonitor_minErrorFlag.value()
    client.BeamClient_makeDiagnosticPlots         = monitor.BeamMonitor_makeDiagnosticPlots.value()
    
    # Dead Cell
    client.DeadCellClient                         = monitor.DeadCellMonitor.value()
    client.DeadCellClient_test_digis              = monitor.DeadCellMonitor_test_digis.value()
    client.DeadCellClient_test_rechits            = monitor.DeadCellMonitor_test_rechits.value()
    client.DeadCellClient_minErrorFlag           = monitor.DeadCellMonitor_minErrorFlag.value() # want to keep these separate?
    client.DeadCellClient_makeDiagnosticPlots     = monitor.DeadCellMonitor_makeDiagnosticPlots.value()         

    # Digi 
    client.DigiClient                             = monitor.DigiMonitor.value()

    # Hot Cell
    client.HotCellClient                          = monitor.HotCellMonitor.value()
    client.HotCellClient_test_persistent          = monitor.HotCellMonitor_test_persistent.value()
    client.HotCellClient_test_energy              = monitor.HotCellMonitor_test_energy.value()
    client.HotCellClient_test_neighbor            = monitor.HotCellMonitor_test_neighbor.value()
    client.HotCellClient_minErrorFlag            = monitor.HotCellMonitor_minErrorFlag.value() # want to keep these separate?
    client.HotCellClient_makeDiagnosticPlots      = monitor.HotCellMonitor_makeDiagnosticPlots.value()

    # Pedestal Client
    client.ReferencePedestalClient                         = monitor.ReferencePedestalMonitor.value()
    client.ReferencePedestalClient_nominalPedMeanInADC     = monitor.ReferencePedestalMonitor_nominalPedMeanInADC.value()
    client.ReferencePedestalClient_nominalPedWidthInADC    = monitor.ReferencePedestalMonitor_nominalPedWidthInADC.value()
    client.ReferencePedestalClient_maxPedMeanDiffADC       = monitor.ReferencePedestalMonitor_maxPedMeanDiffADC.value()
    client.ReferencePedestalClient_maxPedWidthDiffADC      = monitor.ReferencePedestalMonitor_maxPedWidthDiffADC.value()
    client.ReferencePedestalClient_makeDiagnosticPlots     = monitor.ReferencePedestalMonitor_makeDiagnosticPlots.value()
    #client.ReferencePedestalClient_minErrorFlag           = monitor.ReferencePedestalMonitor_minErrorFlag.value() # want to keep these separate?

    # Rec Hit Client
    client.RecHitClient                           = monitor.RecHitMonitor.value()
    client.RecHitClient_minErrorFlag              = monitor.RecHitMonitor_minErrorFlag.value()
    client.RecHitClient_makeDiagnosticPlots       = monitor.RecHitMonitor_makeDiagnosticPlots.value()

    #########################################################################
    # Noise Client
    client.NoiseClient                           = monitor.DetDiagNoiseMonitor
    #########################################################################

    client.DataFormatClient  = monitor.DataFormatMonitor.value()
    client.LEDClient         = monitor.LEDMonitor.value()
    client.CaloTowerClient   = monitor.CaloTowerMonitor.value()
    client.TrigPrimClient    = monitor.TrigPrimMonitor.value()

    client.showTiming        = monitor.showTiming.value()
    client.debug             = monitor.debug.value()

    if (debug):
        print "HcalMonitorClient values from HcalMonitorModule: "
        print "Debug              = ", client.debug
        print "Online             = ", client.Online
        print "Channel status output dir = ", client.databasedir
        print "Nlumiblocks        = ", client.Nlumiblocks
        print "showTiming         = ", client.showTiming
        print "PrescaleEvt        = ", client.diagnosticPrescaleEvt
        print "ReferencePedestal Client    = ", client.ReferencePedestalClient
        print "Digi Client        = ", client.DigiClient
        print "DeadCell Client    = ", client.DeadCellClient
        print "\t\t Min Error Flag  = ",client.DeadCellClient_minErrorFlag
        print "\t\t make diagnostics? ",client.DeadCellClient_makeDiagnosticPlots

        print "HotCell Client    = ", client.HotCellClient
        print "\t\t Test HotCell persistently above threshold? ", client.HotCellClient_test_persistent
        #print "\t\t Test HotCell pedestal? ",                     client.HotCellClient_test_pedestal
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

    #########################################################################
        print "Noise Client      = ",   client.NoiseClient
    #########################################################################

        print "Beam Client      = ",     client.BeamClient
        print "\t\t CheckNevents  = ",   client.BeamClient_checkNevents
        print "\t\t MinErrorFlag  = ",   client.BeamClient_minErrorFlag
        print "\t\t make diagnostics? ", client.BeamClient_makeDiagnosticPlots
        print "CaloTower Client   = ",   client.CaloTowerClient
        print "TrigPrim Client    = ",   client.TrigPrimClient

    return
