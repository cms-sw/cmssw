ecalPresampleTask = dict(
    MEs = dict(
        Pedestal = dict(path = "%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D')
    )
)
