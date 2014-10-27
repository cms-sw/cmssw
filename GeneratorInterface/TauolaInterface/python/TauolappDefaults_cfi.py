import FWCore.ParameterSet.Config as cms

TauolappDefaults = cms.untracked.PSet(
    UseTauolaPolarization = cms.bool(True),
    InputCards = cms.PSet(
        pjak1 = cms.int32(0),
        pjak2 = cms.int32(0),
        mdtau = cms.int32(0)
        ),
    parameterSets = cms.vstring("setTauBr"),
    setTauBr = cms.PSet( # update BR to PDG 2014 BR Ian M. Nugent 10/27/2014
        JAK = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),
        BR  = cms.vdouble(0.1783,         # JAKID 1 tau->enunu
                          0.1741,         # JAKID 2 tau->mununu
                          0.1083,         # JAKID 3 tau->pinu
                          0.2552,         # JAKID 4 tau->pipi0nu
                          0.0930+0.0902,  # JAKID 5 tau->pipipinu/pipi0pi0nu
                          0.0070,         # JAKID 6 tau->Knu
                          0.00429+0.0084, # JAKID 7 tau->K*nu = KS0pi/KL0pi/Kpi0
                          0.0448,         # JAKID 8 tau->3pipi0nu
                          0.0105,  #     JAKID 9 tau->pi3pi0nu
                          0.00498, # JAKID 10 tau->3pi2pi0nu
                          0.00083, # JAKID 11 tau->5pinu
                          0.00016, # JAKID 12 tau->5pipi0nu
                          0.00021, # JAKID 13 tau->3pi3pi0nu
                          0.00144, # JAKID 14 tau->KpiKnu
                          0.0017,  # JAKID 15 tau->K0BpiK0nu
                          0.00159, # JAKID 16 tau->KK0Bpi0nu
                          0.00065, # JAKID 17 tau->K2pi0nu
                          0.00294, # JAKID 18 tau->K2pinu
                          0.0040, # JAKID 19 tau->K0pipi0nu
                          0.00139, # JAKID 20 tau->etapipi0nu
                          0.1083, # JAKID 21 tau->pipi0nugamma
                          0.00159  # JAKID 22 tau->KK0Bnugamma
                          )
        ),
    ),

