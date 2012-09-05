import FWCore.ParameterSet.Config as cms

# Single muon for Wjets
isomuons = cms.EDFilter(
        "MuonSelector",
            src = cms.InputTag('muons'),
        cut = cms.string(    "(isTrackerMuon) && abs(eta) < 2.5 && pt > 9.5"+#17. "+
                             "&& globalTrack.isNonnull"+
                             "&& innerTrack.hitPattern.numberOfValidPixelHits > 0"+
                             "&& innerTrack.normalizedChi2 < 10"+
                             "&& numberOfMatches > 0"+
                             "&& innerTrack.hitPattern.numberOfValidTrackerHits>5"+
                             "&& globalTrack.hitPattern.numberOfValidHits>0"+
                             "&& (isolationR03.sumPt/pt) < 0.2"+
                             #"&& (pfIsolationR03.sumChargedHadronPt)/pt < 0.2"+
                             "&& (pfIsolationR03.sumChargedHadronPt)/pt < 0.2"+
                             "&& abs(innerTrack().dxy)<2.0"
                             ),
        filter = cms.bool(False)
        )

isoelectrons = cms.EDFilter(
    "GsfElectronSelector",
            src = cms.InputTag('gsfElectrons'),
            cut = cms.string(
            "abs(eta) < 2.5 && pt > 9.5"                               +
            "&& gsfTrack.trackerExpectedHitsInner.numberOfHits == 0"   +
            #"&& (pfIsolationVariables.chargedHadronIso)/et  < 0.2"     +
            "&& (isolationVariables03.tkSumPt)/et              < 0.2"  +
            "&& ((abs(eta) < 1.4442  "                                 +
            "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.007"+
            "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.8"  +
            "&& sigmaIetaIeta                                  < 0.01" +
            "&& hcalOverEcal                                   < 0.15" +
            "&& abs(1./superCluster.energy - 1./p)             < 0.05)" +
            "|| (abs(eta)  > 1.566 "+
            "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.007"+
            "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.03" +
            "&& sigmaIetaIeta                                  < 0.03" +
            "&& hcalOverEcal                                   < 0.10" +
            "&& abs(1./superCluster.energy - 1./p)             < 0.05))" 
            ),
        filter = cms.bool(False)
        )

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets as dummy
kt6PFJetsForRhoComputationVoronoi = dummy.clone(
        doRhoFastjet = True,
            voronoiRfact = 0.9
        )

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator
hpsPFTauDiscriminationByDecayModeFinding = hpsSelectionDiscriminator.clone(
        PFTauProducer = cms.InputTag('hpsPFTauProducer')
            )

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack
# Define decay mode prediscriminant
requireDecayMode = cms.PSet(
        BooleanOperator = cms.string("and"),
            decayMode = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
                    cut = cms.double(0.5)
                )
        )

hpsPFTauDiscriminationByMVAIsolation = cms.EDProducer(
    "PFRecoTauDiscriminationByMVAIsolation",
            PFTauProducer = cms.InputTag('hpsPFTauProducer'),
            rhoProducer = cms.InputTag('kt6PFJetsForRhoComputationVoronoi','rho'),
            Prediscriminants = requireDecayMode.clone(),
            gbrfFilePath = cms.FileInPath('RecoTauTag/RecoTau/data/gbrfTauIso.root'),
            returnMVA = cms.bool(False),
            mvaMin = cms.double(0.7),
            )

isotaus = cms.EDFilter(
    "PFTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    BooleanOperator = cms.string("and"),
    discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),       selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByMVAIsolation"),           selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseElectronRejection"), selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"),     selectionCut=cms.double(0.5)) 
    ),
    cut = cms.string("abs(eta) < 2.3 && pt > 19.0 "),
    filter = cms.bool(False)
    )

isomuonseq     = cms.Sequence(isomuons)
isoelectronseq = cms.Sequence(isoelectrons)
isotauseq      = cms.Sequence(
     kt6PFJetsForRhoComputationVoronoi*
     hpsPFTauDiscriminationByMVAIsolation*
     isotaus
    )

leptonSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring(
    'isomuonseq',
    'isoelectronseq',
    'isotauseq')
    )
    )
