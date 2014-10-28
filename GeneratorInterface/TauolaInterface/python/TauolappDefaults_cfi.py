import FWCore.ParameterSet.Config as cms

# relocate PSet definitions from GeneratorInterface/ExternalDecays/python/TauolaSettings_cff.py 
TauolaDefaultInputCards = cms.PSet(
    InputCards = cms.PSet(
        pjak1 = cms.int32(0),
        pjak2 = cms.int32(0),
        mdtau = cms.int32(0)
        )
    )

TauolaNoPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(False)
    )

TauolaPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(True)
    )

# Define BR setups 
TauolaBRDefault = cms.PSet( 
    parameterSets = cms.vstring("setTauBr"),
    setTauBr = cms.PSet(
        # Update BR to PDG 2014 BR Ian M. Nugent 10/27/2014
        # GX is the \Gamma_{X} from the PDG labelling
        #(http://pdg8.lbl.gov/rpp2014v1/pdgLive/Particle.action?node=S035#decayclump_C)
        # Note: Total BR is 0.995575 not 1.0 due to missing modes - Tauola automatically
        #       rescales to 1.0.
        JAK = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),
        BR  = cms.vdouble(0.1783,         # G3      JAKID 1 tau->enunu
                          0.1741,         # G5      JAKID 2 tau->mununu
                          0.1083,         # G9      JAKID 3 tau->pinu
                          0.2536,         # G14     JAKID 4 tau->pipi0nu
                          0.1832,         # G20+G62 JAKID 5 tau->pipipinu/pipi0pi0nu
                          0.0070,         # G10     JAKID 6 tau->Knu
                          0.01269,        # G16+G35 JAKID 7 tau->K*nu = KS0pi/KL0pi/Kpi0
                          0.0448,         # G71     JAKID 8 tau->3pipi0nu
                          0.0105,         # G27     JAKID 9 tau->pi3pi0nu
                          0.00498,        # G78     JAKID 10 tau->3pi2pi0nu
                          0.00083,        # G107    JAKID 11 tau->5pinu
                          0.000165,       # G114    JAKID 12 tau->5pipi0nu
                          0.00021,        # G81     JAKID 13 tau->3pi3pi0nu
                          0.00144,        # G98     JAKID 14 tau->KpiKnu
                          0.0017,         # G46     JAKID 15 tau->K0BpiK0nu
                          0.00159,        # G42     JAKID 16 tau->KK0Bpi0nu
                          0.00065,        # G23     JAKID 17 tau->K2pi0nu
                          0.00294,        # G90     JAKID 18 tau->K2pinu
                          0.0040,         # G40     JAKID 19 tau->K0pipi0nu
                          0.00139,        # G140    JAKID 20 tau->etapipi0nu
                          0.00160,        # Guess   JAKID 21 tau->pipi0nugamma Guess from Tauolapp original settings
                          0.00159         # G37 JAKID 22 tau->KK0Bnugamma
                          )
        )
    )

TauolaBRLEP = cms.PSet(
    parameterSets = cms.vstring("setTauBr"),
    setTauBr = cms.PSet(
        # Update BR to PDG 2014 BR Ian M. Nugent 10/27/2014
        # GX is the \Gamma_{X} from the PDG labelling
        #(http://pdg8.lbl.gov/rpp2014v1/pdgLive/Particle.action?node=S035#decayclump_C)
        # Note: Set all hadronic BR=0
        JAK = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),
        BR  = cms.vdouble(0.1783,         # G3      JAKID 1 tau->enunu
                          0.1741,         # G5      JAKID 2 tau->mununu
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          )
        )
    )

TauolaBRHAD = cms.PSet(
    parameterSets = cms.vstring("setTauBr"),
    setTauBr = cms.PSet(
        # Update BR to PDG 2014 BR Ian M. Nugent 10/27/2014
        # GX is the \Gamma_{X} from the PDG labelling
        #(http://pdg8.lbl.gov/rpp2014v1/pdgLive/Particle.action?node=S035#decayclump_C)
        # Note: Set all leptonic BR=0
        JAK = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),
        BR  = cms.vdouble(0.0,
                          0.0,
                          0.1083,         # G9      JAKID 3 tau->pinu
                          0.2536,         # G14     JAKID 4 tau->pipi0nu
                          0.1832,         # G20+G62 JAKID 5 tau->pipipinu/pipi0pi0nu
                          0.0070,         # G10     JAKID 6 tau->Knu
                          0.01269,        # G16+G35 JAKID 7 tau->K*nu = KS0pi/KL0pi/Kpi0
                          0.0448,         # G71     JAKID 8 tau->3pipi0nu
                          0.0105,         # G27     JAKID 9 tau->pi3pi0nu
                          0.00498,        # G78     JAKID 10 tau->3pi2pi0nu
                          0.00083,        # G107    JAKID 11 tau->5pinu
                          0.000165,       # G114    JAKID 12 tau->5pipi0nu
                          0.00021,        # G81     JAKID 13 tau->3pi3pi0nu
                          0.00144,        # G98     JAKID 14 tau->KpiKnu
                          0.0017,         # G46     JAKID 15 tau->K0BpiK0nu
                          0.00159,        # G42     JAKID 16 tau->KK0Bpi0nu
                          0.00065,        # G23     JAKID 17 tau->K2pi0nu
                          0.00294,        # G90     JAKID 18 tau->K2pinu
                          0.0040,         # G40     JAKID 19 tau->K0pipi0nu
                          0.00139,        # G140    JAKID 20 tau->etapipi0nu
                          0.00160,        # Guess   JAKID 21 tau->pipi0nugamma Guess from Tauolapp original settings
                          0.00159         # G37 JAKID 22 tau->KK0Bnugamma
                          )
        )
    )

#Define default setups for Tauolapp
TauolappDefaults = cms.untracked.PSet(
    TauolaPolar,
    TauolaDefaultInputCards,
    TauolaBRDefault
    )

TauolappLeptonic = cms.untracked.PSet(
    TauolaPolar,
    TauolaDefaultInputCards,
    TauolaBRLEP
    )

TauolappAllHadronic = cms.untracked.PSet(
    TauolaPolar,
    TauolaDefaultInputCards,
    TauolaBRHAD
    )

