import FWCore.ParameterSet.Config as cms

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
        L1ASRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and SRP.')
        ),
        FEDFatal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/FEDIntegrity/FEDFatal'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('')
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
        BXDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.')
        ),
        DesyncByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
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
            description = cms.untracked.string('Total number of front-ends in error status in this lumi section.')
        ),
        TrendNSyncErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/RawDataTask accumulated number of sync errors'),
            kind = cms.untracked.string('TH1F'),
            cumulative = cms.untracked.bool(True),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Accumulated trend of the number of synchronization errors (L1A & BX mismatches) between DCC and FE in this run.')
        ),
        EventTypePostCalib = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0.0)
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
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing < 3490')
        ),
        FEDEntries = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/FEDIntegrity/FEDEntries'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('')
        ),
        EventTypeCalib = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing == 3490. This plot is filled using data from the physics data stream during physics runs. It is normal to have very few entries in these cases.')
        ),
        Entries = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT raw data entries'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Number of events recorded by each FED.')
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
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(0.0)
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('FE status counter.')
        )
    )
)
