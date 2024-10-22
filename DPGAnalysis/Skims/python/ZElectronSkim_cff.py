import FWCore.ParameterSet.Config as cms
# run on MIONAOD
RUN_ON_MINIAOD = False


# cuts
ELECTRON_CUT=("pt > 10 && abs(eta)<2.5")

# single lepton selectors
if RUN_ON_MINIAOD:
    goodZeeElectrons = cms.EDFilter("PATElectronRefSelector",
                                    src = cms.InputTag("slimmedElectrons"),
                                    cut = cms.string(ELECTRON_CUT)
                                    )
else:
    goodZeeElectrons = cms.EDFilter("GsfElectronRefSelector",
                                    src = cms.InputTag("gedGsfElectrons"),
                                    cut = cms.string(ELECTRON_CUT)
                                    )

eleIDWP = cms.PSet( #first for barrel, second for endcap. All values from https://indico.cern.ch/event/699197/contributions/2900013/attachments/1604361/2544765/Zee.pdf
    full5x5_sigmaIEtaIEtaCut       = cms.vdouble(0.0128 ,0.0445 )  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = cms.vdouble(0.00523,0.00984)  , # dEtaInSeedCut
    dPhiInCut                      = cms.vdouble(0.159  ,0.157  )  , # dPhiInCut
    hOverECut                      = cms.vdouble(0.247  ,0.0982  )  , # hOverECut
    relCombIsolationWithEACut      = cms.vdouble(0.168  ,0.185  )  , # relCombIsolationWithEALowPtCut
    EInverseMinusPInverseCut       = cms.vdouble(0.193  ,0.0962   )  ,                
    missingHitsCut                 = cms.vint32(2       ,3      )    # missingHitsCut
) 


identifiedElectrons = cms.EDFilter("ZElectronsSelectorAndSkim",
                                   src    = cms.InputTag("goodZeeElectrons"),
                                   eleID = eleIDWP, 
                                   absEtaMin=cms.vdouble( 0.0000, 1.0000, 1.4790, 2.0000, 2.2000, 2.3000, 2.4000),
                                   absEtaMax=cms.vdouble( 1.0000,  1.4790, 2.0000,  2.2000, 2.3000, 2.4000, 5.0000),
                                   effectiveAreaValues=cms.vdouble( 0.1703, 0.1715, 0.1213, 0.1230, 0.1635, 0.1937, 0.2393),
                                   rho = cms.InputTag("fixedGridRhoFastjetCentralCalo") #from https://github.com/cms-sw/cmssw/blob/09c3fce6626f70fd04223e7dacebf0b485f73f54/RecoEgamma/ElectronIdentification/python/Identification/cutBasedElectronID_tools.py#L564
                         )
DIELECTRON_CUT=("mass > 40  && daughter(0).pt>20 && daughter(1).pt()>10")

diZeeElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                decay       = cms.string("identifiedElectrons identifiedElectrons"),
                                checkCharge = cms.bool(False),
                                cut         = cms.string(DIELECTRON_CUT)
                                )
# dilepton counters
diZeeElectronsFilter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("diZeeElectrons"),
                                    minNumber = cms.uint32(1)
                                    )


#sequences
zdiElectronSequence = cms.Sequence(goodZeeElectrons*identifiedElectrons*diZeeElectrons* diZeeElectronsFilter )
