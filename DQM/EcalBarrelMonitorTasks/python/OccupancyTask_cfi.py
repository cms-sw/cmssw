occupancyTask = dict(
    recHitThreshold = 0.5,
    tpThreshold = 4.,
    MEs = dict(
        Digi = dict(path = "Occupancy/Digi/OccupancyTask digi occupancy", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        DigiProjEta = dict(path = "Occupancy/Digi/OccupancyTask digi occupancy", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        DigiProjPhi = dict(path = "Occupancy/Digi/OccupancyTask digi occupancy", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        DigiAll = dict(path = "Occupancy/Digi/OccupancyTask digi occupancy", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        DigiDCC = dict(path = "Occupancy/Digi/OccupancyTask digi occupancy by DCC", otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        RecHit1D = dict(path = "Occupancy/RecHit/OccupancyTask recHit number", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 6000.}),
        RecHitThr = dict(path = "Occupancy/RecHitThres/OccupancyTask recHit thres occupancy", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        RecHitThrProjEta = dict(path = "Occupancy/RecHitThres/OccupancyTask recHit thres occupancy", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        RecHitThrProjPhi = dict(path = "Occupancy/RecHitThres/OccupancyTask recHit thres occupancy", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        RecHitThrAll = dict(path = "Occupancy/RecHitThres/OccupancyTask recHit thres occupancy", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        TPDigi = dict(path = "Occupancy/TPDigi/OccupancyTask TP digi occupancy", otype = 'SM', btype = 'TriggerTower', kind = 'TH2F'),
        TPDigiProjEta = dict(path = "Occupancy/TPDigi/OccupancyTask TP digi occupancy", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        TPDigiProjPhi = dict(path = "Occupancy/TPDigi/OccupancyTask TP digi occupancy", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        TPDigiThr = dict(path = "Occupancy/TPDigiThres/OccupancyTask TP digi thres occupancy", otype = 'SM', btype = 'TriggerTower', kind = 'TH2F'),
        TPDigiThrProjEta = dict(path = "Occupancy/TPDigiThres/OccupancyTask TP digi thres occupancy", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TH1F'),
        TPDigiThrProjPhi = dict(path = "Occupancy/TPDigiThres/OccupancyTask TP digi thres occupancy", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TH1F'),
        TPDigiThrAll = dict(path = "Occupancy/TPDigiThres/OccupancyTask TP digi thres occupancy", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F')
    )
)
