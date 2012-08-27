towerStatusTask = dict(
    doDAQInfo = True,
    doDCSInfo = True,
    MEs = dict(
        DAQSummary = dict(path = "EventInfo/DAQSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL'),
        DAQSummaryMap = dict(path = "EventInfo/DAQSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        DAQContents = dict(path = "EventInfo/DAQContents/", otype = 'SM', btype = 'Report', kind = 'REAL'),
        DCSSummary = dict(path = "EventInfo/DCSSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL'),
        DCSSummaryMap = dict(path = "EventInfo/DCSSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        DCSContents = dict(path = "EventInfo/DCSContents/", otype = 'SM', btype = 'Report', kind = 'REAL')
    )
)
