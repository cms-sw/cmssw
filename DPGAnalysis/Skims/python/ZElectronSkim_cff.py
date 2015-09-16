import FWCore.ParameterSet.Config as cms

# run on MIONAOD
RUN_ON_MINIAOD = False
#print "ZEE SKIM. RUN ON MINIAOD = ",RUN_ON_MINIAOD

# cuts
ELECTRON_CUT=("obj.pt() > 10 && std::abs(obj.eta())<2.5")
DIELECTRON_CUT=("obj.mass() > 40 && obj.mass() < 140 && obj.daughter(0)->pt()>20 && obj.daughter(1)->pt()>10")


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
                                                              " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)<=2)"
                                                              " && ((obj.isEB()"
                                                              " && ( ((obj.pfIsolationVariables().sumChargedHadronPt + std::max(0.0,obj.pfIsolationVariables().sumNeutralHadronEt + obj.pfIsolationVariables().sumPhotonEt - 0.5 * obj.pfIsolationVariables().sumPUPt))/obj.p4().pt())<0.164369)"
                                                              " && (obj.full5x5_sigmaIetaIeta()<0.011100)"
                                                              " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx()) < 0.252044 )"
                                                              " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx()) < 0.016315 )"
                                                              " && (obj.hadronicOverEm()<0.345843)"
                                                              ")"
                                                              " || (obj.isEE()"
                                                              " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)<=3)"
                                                              " && ( ((obj.pfIsolationVariables().sumChargedHadronPt + std::max(0.0,obj.pfIsolationVariables().sumNeutralHadronEt + obj.pfIsolationVariables().sumPhotonEt - 0.5 * obj.pfIsolationVariables().sumPUPt))/obj.p4().pt())<0.212604 )"
                                                              " && (obj.full5x5_sigmaIetaIeta()<0.033987)"
                                                              " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.245263 )"
                                                              " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.010671 )"
                                                              " && (obj.hadronicOverEm()<0.134691) "
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
