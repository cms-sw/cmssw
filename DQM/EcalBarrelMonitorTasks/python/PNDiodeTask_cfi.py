pnDiodeTask = dict(
    MEs = dict(
        MEMChId = dict(path = 'PN/Integrity/MEMChId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMGain = dict(path = 'PN/Integrity/MEMGain/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMBlockSize = dict(path = 'PN/Integrity/MEMBlockSize/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        MEMTowerId = dict(path = 'PN/Integrity/MEMTowerId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        Pedestal = dict(path = "PN/Presample/PNDiodeTask PN pedestal G16", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile'),
        Occupancy = dict(path = "PN/Occupancy/PNDiodeTask MEM digi occupancy", otype = "SMMEM", btype = 'Crystal', kind = 'TH1F')
    )
)
