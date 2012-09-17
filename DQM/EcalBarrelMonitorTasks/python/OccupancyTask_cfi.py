ecalOccupancyTask = dict(
    recHitThreshold = 0.5,
    tpThreshold = 4.,
    MEs = dict(
        Digi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        DigiProjEta = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection eta", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        DigiProjPhi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection phi", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        DigiAll = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        DigiDCC = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy by DCC", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Digi1D = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of digis in event", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 200, 'low': 0., 'high': 3000.}),
        RecHitAll = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s", otype = 'Ecal3P', btype = 'Crystal', kind = 'TH2F'),
        RecHitProjEta = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection eta", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        RecHitProjPhi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection phi", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        RecHitThrProjEta = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection eta", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        RecHitThrProjPhi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection phi", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        RecHitThrAll = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        RecHitThr1D = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of filtered rec hits in event", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 200, 'low': 0., 'high': 500.}),
        TPDigiProjEta = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi occupancy%(suffix)s projection eta", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        TPDigiProjPhi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi occupancy%(suffix)s projection phi", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        TPDigiAll = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi occupancy%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        TPDigiThrProjEta = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection eta", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        TPDigiThrProjPhi = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection phi", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        TPDigiThrAll = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        TrendNDigi = dict(path = "Ecal/Trends/OccupancyTask %(prefix)s number of digis", otype = 'Ecal2P', btype = 'Trend', kind = 'TProfile'),
        TrendNRecHitThr = dict(path = 'Ecal/Trends/OccupancyTask %(prefix)s number of filtered recHits', otype = 'Ecal2P', btype = 'Trend', kind = 'TProfile'),
        TrendNTPDigi = dict(path = 'Ecal/Trends/OccupancyTask %(prefix)s number of filtered TP digis', otype = 'Ecal2P', btype = 'Trend', kind = 'TProfile')
    )
)
