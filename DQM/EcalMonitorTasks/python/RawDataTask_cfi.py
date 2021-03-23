import FWCore.ParameterSet.Config as cms

eventTypes = [
    "UNKNOWN",
    "COSMIC",
    "BEAMH4",
    "BEAMH2",
    "MTCC",
    "LASER_STD",
    "LASER_POWER_SCAN",
    "LASER_DELAY_SCAN",
    "TESTPULSE_SCAN_MEM",
    "TESTPULSE_MGPA",
    "PEDESTAL_STD",
    "PEDESTAL_OFFSET_SCAN",
    "PEDESTAL_25NS_SCAN",
    "LED_STD",
    "PHYSICS_GLOBAL",
    "COSMICS_GLOBAL",
    "HALO_GLOBAL",
    "LASER_GAP",
    "TESTPULSE_GAP",
    "PEDESTAL_GAP",
    "LED_GAP",
    "PHYSICS_LOCAL",
    "COSMICS_LOCAL",
    "HALO_LOCAL",
    "CALIB_LOCAL"
]

statuses = [
    "ENABLED",
    "DISABLED",
    "TIMEOUT",
    "HEADERERROR",
    "CHANNELID",
    "LINKERROR",
    "BLOCKSIZE",
    "SUPPRESSED",
    "FIFOFULL",
    "L1ADESYNC",
    "BXDESYNC",
    "L1ABXDESYNC",
    "FIFOFULLL1ADESYNC",
    "HPARITY",
    "VPARITY",
    "FORCEDZS"
]

ecalRawDataTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        BXSRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing SRP errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and SRP.')
        ),
        CRC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT CRC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of CRC errors.')
        ),
        BXFE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE errors'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.')
        ),
        BXFEDiff = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE-DCC'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.')
        ),
        BXFEInvalid = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing invalid value'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(69.0),
                nbins = cms.untracked.int32(69),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.')
        ),
        L1ASRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and SRP.')
        ),
        BXTCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing TCC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of bunch corssing value mismatches between DCC and TCC.')
        ),
        DesyncTotal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT total FE synchronization errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE.')
        ),
        RunNumber = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT run number errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between run numbers recorded in the DCC and that in CMS Event.')
        ),
        Orbit = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.')
        ),
        OrbitDiff = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number DCC-GT'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.)
            ),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.')
        ),
        BXDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.')
        ),
        BXDCCDiff = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC-GT'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.)
            ),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.')
        ),
        DesyncByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            perLumi = cms.untracked.bool(True),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.')
        ),
        L1ATCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A TCC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and TCC.')
        ),
        FEByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            perLumi = cms.untracked.bool(True),
            description = cms.untracked.string('Total number of front-ends in error status in this lumi section.')
        ),
        TrendNSyncErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/RawDataTask accumulated number of sync errors'),
            kind = cms.untracked.string('TH1F'),
            cumulative = cms.untracked.bool(True),
            online = cms.untracked.bool(True),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Accumulated trend of the number of synchronization errors (L1A & BX mismatches) between DCC and FE in this run.')
        ),
        EventTypePostCalib = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(eventTypes)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing > 3490.')
        ),
        L1ADCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.')
        ),
        EventTypePreCalib = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type pre calibration BX'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(eventTypes)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing < 3490')
        ),
        EventTypeCalib = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(eventTypes)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing == 3490. This plot is filled using data from the physics data stream during physics runs. It is normal to have very few entries in these cases.')
        ),
        L1AFE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A FE errors'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and FE.')
        ),
        TriggerType = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT trigger type errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between trigger type recorded in the DCC and that in CMS Event.')
        ),
        FEStatus = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(statuses)
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('FE status counter.')
        ),
        FEStatusErrMapByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT%(suffix)s front-end status error map by lumi'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            perLumi = cms.untracked.bool(True),
            description = cms.untracked.string('FE status error occupancy map for this lumisection. Nominal FE status flags such as ENABLED, SUPPRESSED, FIFOFILL, FIFOFULLL1ADESYNC, and FORCEDZS are NOT included.')
        ),
        FEStatusMEM = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/MEM/StatusFlagsTask MEM front-end status bits'),
            kind = cms.untracked.string('TH2F'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(55),
                nbins = cms.untracked.int32(108),
                low = cms.untracked.double(1),
            ),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(statuses)
            ),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Front-end (FE) status counter for MEM boxes. Each x-axis tick corresponds to one SuperModule (SM) as indexed by DCC Id and contains two bins corresponding to the MEM boxes (DCC tower Ids = 69, 70). Nominal status is SUPPRESSED. EE+/-2,3,7,8 are not connected to MEM boxes and instead appear with status DISABLED. Mapping from DCC Id to SM name appears below.<br/><pre>01:EE-07  19:EB-10  37:EB+10<br/>02:EE-08  20:EB-11  38:EB+11<br/>03:EE-09  21:EB-12  39:EB+12<br/>04:EE-01  22:EB-13  40:EB+13<br/>05:EE-02  23:EB-14  41:EB+14<br/>06:EE-03  24:EB-15  42:EB+15<br/>07:EE-04  25:EB-16  43:EB+16<br/>08:EE-05  26:EB-17  44:EB+17<br/>09:EE-06  27:EB-18  45:EB+18<br/>10:EB-01  28:EB+01  46:EE+07<br/>11:EB-02  29:EB+02  47:EE+08<br/>12:EB-03  30:EB+03  48:EE+09<br/>13:EB-04  31:EB+04  49:EE+01<br/>14:EB-05  32:EB+05  50:EE+02<br/>15:EB-06  33:EB+06  51:EE+03<br/>16:EB-07  34:EB+07  52:EE+04<br/>17:EB-08  35:EB+08  53:EE+05<br/>18:EB-09  36:EB+09  54:EE+06</pre>')
        )
    )
)
