import FWCore.ParameterSet.Config as cms
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
IsoPhotonEBHLTFilter = copy.deepcopy(hltHighLevel)
IsoPhotonEBHLTFilter.throw = cms.bool(False)
IsoPhotonEBHLTFilter.HLTPaths = ["HLT_Photon110EB_TightID_TightIso_v*"]

# run on MIONAOD
RUN_ON_MINIAOD = False


# cuts
PHOTON_CUT=("pt > 110 && abs(eta)<1.4442")

# single lepton selectors
if RUN_ON_MINIAOD:
    goodPhotons = cms.EDFilter("PATElectronRefSelector",
                                    src = cms.InputTag("slimmedPhotons"),
                                    cut = cms.string(PHOTON_CUT)
                                    )
else:
    goodPhotons = cms.EDFilter("PhotonRefSelector",
                                    src = cms.InputTag("gedPhotons"),
                                    cut = cms.string(PHOTON_CUT)
                                    )

photonIDWP = cms.PSet( #first for barrel, second for endcap. 
    full5x5_sigmaIEtaIEtaCut       = cms.vdouble(0.011 ,-1. )  , # full5x5_sigmaIEtaIEtaCut
    hOverECut                      = cms.vdouble(0.1  ,-1. )  , # hOverECut
    relCombIsolationWithEACut      = cms.vdouble(0.05  ,-1. )   # relCombIsolationWithEALowPtCut
) 


identifiedPhotons = cms.EDFilter("IsoPhotonEBSelectorAndSkim",
                                   src    = cms.InputTag("goodPhotons"),
                                   phID = photonIDWP, 
                                   absEtaMin=cms.vdouble( 0.0000, 1.0000, 1.4790, 2.0000, 2.2000, 2.3000, 2.4000),
                                   absEtaMax=cms.vdouble( 1.0000,  1.4790, 2.0000,  2.2000, 2.3000, 2.4000, 5.0000),
                                   effectiveAreaValues=cms.vdouble( 0.1703, 0.1715, 0.1213, 0.1230, 0.1635, 0.1937, 0.2393),
                                   rho = cms.InputTag("fixedGridRhoFastjetCentralCalo") 
                         )

identifiedPhotonsCountFilter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("identifiedPhotons"),
                                    minNumber = cms.uint32(1)
                                    )



#sequences
isoPhotonEBSequence = cms.Sequence(IsoPhotonEBHLTFilter*goodPhotons*identifiedPhotons*identifiedPhotonsCountFilter )
