ecalPnDiodeTask = dict(
    MEs = dict(
        MEMChId = dict(path = 'Ecal/Errors/Integrity/MEMChId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMGain = dict(path = 'Ecal/Errors/Integrity/MEMGain/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMBlockSize = dict(path = 'Ecal/Errors/Integrity/MEMBlockSize/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMTowerId = dict(path = 'Ecal/Errors/Integrity/MEMTowerId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        Pedestal = dict(path = "%(subdet)s/%(prefix)sPedestalOnlineTask/PN/%(prefix)sPOT PN pedestal %(sm)s G16", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile'),
        Occupancy = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s", otype = "SMMEM", btype = 'Crystal', kind = 'TH1F'),
        OccupancySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT PN digi occupancy summary", otype = "Ecal2P", btype = 'Crystal', kind = 'TH2F')
    )
)
