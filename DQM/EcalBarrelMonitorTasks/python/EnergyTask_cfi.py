ecalEnergyTask = dict(
    isPhysicsRun = True,
    threshS9 = 0.125,
    MEs = dict(
        HitMap = dict(path = 'Energy/Profile/EnergyTask recHit profile', otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', zaxis = {'title': 'energy (GeV)'}),
        HitMapAll = dict(path = 'Energy/Profile/EnergyTask recHit profile', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TProfile2D', zaxis = {'title': 'energy (GeV)'}),
        Hit = dict(path = 'Energy/Spectrum/Crystal/EnergyTask recHit', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 20., 'title': 'energy (GeV)'}),
        HitAll = dict(path = 'Energy/Spectrum/EnergyTask recHit', otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 20., 'title': 'energy (GeV)'}),
#        MiniCluster = dict(path = 'Energy/Spectrum/3x3/EnergyTask 3x3', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 20., 'title': 'energy (GeV)'})
    )
)

    
