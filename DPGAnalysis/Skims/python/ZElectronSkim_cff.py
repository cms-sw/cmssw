import FWCore.ParameterSet.Config as cms

# run on MIONAOD
RUN_ON_MINIAOD = False
#print "ZEE SKIM. RUN ON MINIAOD = ",RUN_ON_MINIAOD

# cuts
ELECTRON_CUT=("pt > 10 && abs(eta)<2.5")
DIELECTRON_CUT=("mass > 40 && mass < 140 && daughter(0).pt>20 && daughter(1).pt()>10")


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

# electron ID (sync with the AlCaReco: https://raw.githubusercontent.com/cms-sw/cmssw/CMSSW_7_5_X/Calibration/EcalAlCaRecoProducers/python/WZElectronSkims_cff.py)
identifiedElectrons = goodZeeElectrons.clone(cut = cms.string(goodZeeElectrons.cut.value() +
                                                              " && (gsfTrack.hitPattern().numberOfHits(\'MISSING_INNER_HITS\')<=2)"
                                                              " && ((isEB"
                                                              " && ( ((pfIsolationVariables().sumChargedHadronPt + max(0.0,pfIsolationVariables().sumNeutralHadronEt + pfIsolationVariables().sumPhotonEt - 0.5 * pfIsolationVariables().sumPUPt))/p4.pt)<0.164369)"
                                                              " && (full5x5_sigmaIetaIeta<0.011100)"
                                                              " && ( - 0.252044<deltaPhiSuperClusterTrackAtVtx< 0.252044 )"
                                                       " && ( -0.016315<deltaEtaSuperClusterTrackAtVtx<0.016315 )"
                                                              " && (hadronicOverEm<0.345843)"
                                                              ")"
                                                              " || (isEE"
                                                              " && (gsfTrack.hitPattern().numberOfHits(\'MISSING_INNER_HITS\')<=3)"
                                                              " && ( ((pfIsolationVariables().sumChargedHadronPt + max(0.0,pfIsolationVariables().sumNeutralHadronEt + pfIsolationVariables().sumPhotonEt - 0.5 * pfIsolationVariables().sumPUPt))/p4.pt)<0.212604 )"
                                                              " && (full5x5_sigmaIetaIeta<0.033987)"
                                                              " && ( -0.245263<deltaPhiSuperClusterTrackAtVtx<0.245263 )"
                                                              " && ( -0.010671<deltaEtaSuperClusterTrackAtVtx<0.010671 )"
                                                              " && (hadronicOverEm<0.134691) "
                                                              "))"
                                                              )
                                             )

# dilepton selectors
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
zdiElectronSequence = cms.Sequence( goodZeeElectrons * identifiedElectrons * diZeeElectrons * diZeeElectronsFilter )
