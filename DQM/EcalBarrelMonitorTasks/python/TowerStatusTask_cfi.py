ecalTowerStatusTask = dict(
    doDAQInfo = True,
    doDCSInfo = True,
    MEs = dict(
        DAQSummary = dict(path = "Ecal/EventInfo/DAQSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL'),
        DAQSummaryMap = dict(path = "Ecal/EventInfo/DAQSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        DAQContents = dict(path = "Ecal/EventInfo/DAQContents/Ecal_%(sm)s", otype = 'SM', btype = 'Report', kind = 'REAL'),
        DCSSummary = dict(path = "Ecal/EventInfo/DCSSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL'),
        DCSSummaryMap = dict(path = "Ecal/EventInfo/DCSSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        DCSContents = dict(path = "Ecal/EventInfo/DCSContents/Ecal_%(sm)s", otype = 'SM', btype = 'Report', kind = 'REAL')
    )
)
