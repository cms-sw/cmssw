rawDataTask = dict(
    MEs = dict(
        EventTypePreCalib = dict(path = "RawData/RawDataTask event type BX lt 3490", otype = 'Ecal', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        EventTypeCalib = dict(path = "RawData/RawDataTask event type BX eq 3490", otype = 'Ecal', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        EventTypePostCalib = dict(path = "RawData/RawDataTask event type BX gt 3490", otype = 'Ecal', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 25, 'low': 0., 'high': 25.}),
        CRC = dict(path = "RawData/RawDataTask CRC errors", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        RunNumber = dict(path = "RawData/RawDataTask DCC-GT run mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Orbit = dict(path = "RawData/RawDataTask DCC-GT orbit mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        TriggerType = dict(path =  "RawData/RawDataTask DCC-GT trigType mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1ADCC = dict(path = "RawData/RawDataTask DCC-GT L1A mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1AFE = dict(path = "RawData/RawDataTask FE-DCC L1A mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 68, 'low': 0., 'high': 68., 'title': 'iFE'}),
        L1ATCC = dict(path = "RawData/RawDataTask TCC-DCC L1A mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        L1ASRP = dict(path = "RawData/RawDataTask SRP-DCC L1A mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXDCC = dict(path = "RawData/RawDataTask DCC-GT BX mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXFE = dict(path = "RawData/RawDataTask FE-DCC BX mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 68, 'low': 0., 'high': 68., 'title': 'iFE'}),
        BXTCC = dict(path = "RawData/RawDataTask TCC-DCC BX mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        BXSRP = dict(path = "RawData/RawDataTask SRP-DCC BX mismatch", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        DesyncByLumi = dict(path = "RawData/RawDataTask sync errors by lumi", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        DesyncTotal = dict(path = "RawData/RawDataTask sync errors total", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEStatus = dict(path = "RawData/FEStatus/RawDataTask FE status", otype = 'SM', btype = 'SuperCrystal', kind = 'TH2F', yaxis = {'nbins': 16, 'low': 0., 'high': 16.}),
        FEByLumi = dict(path = "RawData/RawDataTask FE status errors by lumi", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEDEntries = dict(path = '%(hlttask)s/FEDEntries', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        FEDFatal = dict(path = '%(hlttask)s/FEDFatal', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        TrendNSyncErrors = dict(path = 'Trend/RawDataTask accumulated number of sync errors', otype = 'Ecal', btype ='Trend', kind = 'TH1F', cumulative = True)
    )
)

