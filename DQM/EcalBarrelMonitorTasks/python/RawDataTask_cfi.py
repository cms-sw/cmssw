ecalRawDataTask = dict(
    MEs = dict(
        EventTypePreCalib = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type pre calibration BX", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        EventTypeCalib = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        EventTypePostCalib = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        CRC = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT CRC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        RunNumber = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT run number errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Orbit = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        TriggerType = dict(path =  "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT trigger type errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1ADCC = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1AFE = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A FE errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 68, 'low': 0., 'high': 68., 'title': 'iFE'}),
        L1ATCC = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A TCC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1ASRP = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXDCC = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXFE = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 68, 'low': 0., 'high': 68., 'title': 'iFE'}),
        BXTCC = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing TCC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXSRP = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing SRP errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        DesyncByLumi = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        DesyncTotal = dict(path = "%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT total FE synchronization errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEStatus = dict(path = "%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TH2F', yaxis = {'nbins': 16, 'low': 0., 'high': 16.}),
        FEByLumi = dict(path = "%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEDEntries = dict(path = '%(subdet)s/FEDIntegrity/FEDEntries', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEDFatal = dict(path = '%(subdet)s/FEDIntegrity/FEDFatal', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        TrendNSyncErrors = dict(path = 'Ecal/Trends/RawDataTask accumulated number of sync errors', otype = 'Ecal', btype ='Trend', kind = 'TH1F', cumulative = True)
    )
)

