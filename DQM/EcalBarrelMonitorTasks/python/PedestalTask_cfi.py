ecalPedestalTask = dict(
    MEs = dict(
        Occupancy = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT pedestal digi occupancy", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        Pedestal = dict(path = "%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 3),
        PNPedestal = dict(path = "%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)
    
