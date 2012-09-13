ecalPedestalTask = dict(
    MEs = dict(
        Occupancy = dict(path = "Occupancy/PedestalTask digi occupancy G%(gain)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', multi = 3),
        Pedestal = dict(path = "Pedestal/Gain%(gain)s/Profile/PedestalTask pedestal G%(gain)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 3),
        PNPedestal = dict(path = "PN/Pedestal/Gain%(pngain)s/PedestalTask PN pedestal G%(pngain)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)
    
