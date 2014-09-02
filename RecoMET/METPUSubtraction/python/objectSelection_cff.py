import FWCore.ParameterSet.Config as cms

##======================================
## Muons
##======================================

selectedMuons = cms.EDFilter(
    "MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string(    "(isTrackerMuon) && abs(eta) < 2.5 && pt > 19.5"+#17. "+
                         "&& isPFMuon"+
                         "&& globalTrack.isNonnull"+
                         "&& innerTrack.hitPattern.numberOfValidPixelHits > 0"+
                         "&& innerTrack.normalizedChi2 < 10"+
                         "&& numberOfMatches > 0"+
                         "&& innerTrack.hitPattern.numberOfValidTrackerHits>5"+
                         "&& globalTrack.hitPattern.numberOfValidHits>0"+
                         "&& (pfIsolationR03.sumChargedHadronPt+pfIsolationR03.sumNeutralHadronEt+pfIsolationR03.sumPhotonEt)/pt < 0.3"+
                         "&& abs(innerTrack().dxy)<2.0"
                         ),
    filter = cms.bool(False)
    )


##======================================
## Electrons
##======================================


selectedElectrons = cms.EDFilter(
    "GsfElectronSelector",
            src = cms.InputTag('gedGsfElectrons'),
            cut = cms.string(
            "abs(eta) < 2.5 && pt > 19.5"                               +
 #           "&& gsfTrack.trackerExpectedHitsInner.numberOfHits == 0"   +
#            "&& (pfIsolationVariables.chargedHadronIso+pfIsolationVariables.neutralHadronIso)/et     < 0.3"  +
            "&& (isolationVariables03.tkSumPt)/et              < 0.2"  +
            "&& ((abs(eta) < 1.4442  "                                 +
            "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.007"+
            "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.8"  +
            "&& sigmaIetaIeta                                  < 0.01" +
            "&& hcalOverEcal                                   < 0.15" +
            "&& abs(1./superCluster.energy - 1./p)             < 0.05)"+
            "|| (abs(eta)  > 1.566 "+
            "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.009"+
            "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.10" +
            "&& sigmaIetaIeta                                  < 0.03" +
            "&& hcalOverEcal                                   < 0.10" +
            "&& abs(1./superCluster.energy - 1./p)             < 0.05))" 
            ),
        filter = cms.bool(False)
        )


##======================================
## Taus
##======================================

selectedTaus = cms.EDFilter("PFTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
            selectionCut = cms.double(0.5)
        ),
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits'),
            selectionCut = cms.double(0.5)
        )                        
    ),
    cut = cms.string("pt > 20. & abs(eta) < 2.3")                        
)



##======================================
## Photons
##======================================

ph_acc = '(pt >= 19 && abs(eta)<2.5)'
ph_id = 'sigmaIetaIeta<0.03 && hadronicOverEm<0.12'
ph_iso = '(chargedHadronIso + neutralHadronIso + photonIso)/pt < 0.3'

selectedPhotons = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("photons"),
    cut = cms.string(ph_acc+"&&"+ph_id+"&&"+ph_iso)
  )




##======================================
## Jets
##======================================

#pileup jetId applied per default in the process

jet_acc = '(pt >= 30 && abs(eta)<2.5)'
#jet_id = ''

selectedJets = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string(jet_acc)
  )


selectionSequenceForMVANoPUMET = cms.Sequence(
selectedMuons+
selectedElectrons+
selectedTaus+
selectedPhotons+
selectedJets
)
