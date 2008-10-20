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
                          digiErrorFrac             = cms.untracked.double(0.05),
                          CapIdMEAN_ErrThresh       = cms.untracked.double(1.5),
                          CapIdRMS_ErrThresh        = cms.untracked.double(0.25),

                          # Dead Cell Client
                          DeadCellClient            = cms.untracked.bool(True),
                          deadcellErrorFrac         = cms.untracked.double(0.05),

                          # DataFormatClient
                          DataFormatClient          = cms.untracked.bool(True),

                          # Hot Cell Client
                          HotCellClient             = cms.untracked.bool(True),
                          hotcellErrorFrac          = cms.untracked.double(0.05),

                          # Summary Client
                          SummaryClient             = cms.untracked.bool(True),

                          #LED Client
                          LEDClient                 = cms.untracked.bool(True),
                          LEDRMS_ErrThresh          = cms.untracked.double(0.8),
                          LEDMEAN_ErrThresh         = cms.untracked.double(2.25),

                          # RecHit Client
                          RecHitClient              = cms.untracked.bool(True),

                          # CaloTowerClient
                          CaloTowerClient           = cms.untracked.bool(False),

                          # TrigPrimClient
                          TrigPrimClient            = cms.untracked.bool(True),

)



def setHcalClientValuesFromMonitor(client, origmonitor, debug=False):
    # need to make separate copy, or changing client after this call will also change monitor!
    monitor=deepcopy(origmonitor)
    
    #Reads variables from monitor module, and sets the client's copy of those variables to the same value.
    #This way, when you disable the DataFormat Monitor, the DataFormat client is also turned off automatically, etc.
    
    client.PedestalClient    = monitor.PedestalMonitor
    client.PedestalClient_nominalPedMeanInADC    = monitor.PedestalMonitor_nominalPedMeanInADC
    client.PedestalClient_nominalPedWidthInADC   = monitor.PedestalMonitor_nominalPedWidthInADC
    client.PedestalClient_maxPedMeanDiffADC      = monitor.PedestalMonitor_maxPedMeanDiffADC
    client.PedestalClient_maxPedWidthDiffADC     = monitor.PedestalMonitor_maxPedWidthDiffADC
    client.PedestalClient_pedestalsInFC          = monitor.PedestalMonitor_pedestalsInFC
    client.PedestalClient_startingTimeSlice      = monitor.PedestalMonitor_startingTimeSlice
    client.PedestalClient_endingTimeSlice        = monitor.PedestalMonitor_endingTimeSlice
    #client.PedestalClient_minErrorFlag           = monitor.   # want to keep these separate?
    
    client.DigiClient        = monitor.DigiMonitor
    client.DeadCellClient    = monitor.DeadCellMonitor
    client.DataFormatClient  = monitor.DataFormatMonitor
    client.HotCellClient     = monitor.HotCellMonitor
    client.LEDClient         = monitor.LEDMonitor
    client.RecHitClient      = monitor.RecHitMonitor
    client.CaloTowerClient   = monitor.CaloTowerMonitor
    client.TrigPrimClient    = monitor.TrigPrimMonitor

    client.showTiming        = monitor.showTiming
    client.debug             = monitor.debug

    if (debug):
        print "HcalMonitorClient values from HcalMonitorModule: "
        print "Debug              = ", client.debug
        print "showTiming         = ", client.showTiming
        print "Pedestal Client    = ", client.PedestalClient
        print "Digi Client        = ", client.DigiClient
        print "DeadCell Client    = ", client.DeadCellClient
        print "DataFormat Client  = ", client.DataFormatClient
        print "HotCell Client     = ", client.HotCellClient
        print "Summary Client     = ", client.SummaryClient
        print "LED Client         = ", client.LEDClient
        print "RecHit Client      = ", client.RecHitClient
        print "CaloTower Client   = ", client.CaloTowerClient
        print "TrigPrim Client    = ", client.TrigPrimClient

    return
