import FWCore.ParameterSet.Config as cms

# Single muon for Wjets
isomuons = cms.EDFilter(
    "MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string(    "(isTrackerMuon) && std::abs(eta) < 2.5 && pt > 9.5"+#17. "+
                         "&& isPFMuon"+
                         "&& globalTrack.isNonnull"+
                         "&& innerTrack.hitPattern.numberOfValidPixelHits > 0"+
                         "&& innerTrack.normalizedChi2 < 10"+
                         "&& numberOfMatches > 0"+
                         "&& innerTrack.hitPattern.numberOfValidTrackerHits>5"+
                         "&& globalTrack.hitPattern.numberOfValidHits>0"+
                         "&& (pfIsolationR03.sumChargedHadronPt+pfIsolationR03.sumNeutralHadronEt+pfIsolationR03.sumPhotonEt)/pt < 0.3"+
                         "&& std::abs(innerTrack().dxy)<2.0"
                         ),
    filter = cms.bool(False)
    )


isoelectrons = cms.EDFilter(
    "GsfElectronSelector",
            src = cms.InputTag('gsfElectrons'),
            cut = cms.string(
            "std::abs(eta) < 2.5 && pt > 9.5"                               +
            "&& gsfTrack.trackerExpectedHitsInner.numberOfHits == 0"   +
#            "&& (pfIsolationVariables.chargedHadronIso+pfIsolationVariables.neutralHadronIso)/et     < 0.3"  +
            "&& (isolationVariables03.tkSumPt)/et              < 0.2"  +
            "&& ((std::abs(eta) < 1.4442  "                                 +
            "&& std::abs(deltaEtaSuperClusterTrackAtVtx)            < 0.007"+
            "&& std::abs(deltaPhiSuperClusterTrackAtVtx)            < 0.8"  +
            "&& sigmaIetaIeta                                  < 0.01" +
            "&& hcalOverEcal                                   < 0.15" +
            "&& std::abs(1./superCluster.energy - 1./p)             < 0.05)"+
            "|| (std::abs(eta)  > 1.566 "+
            "&& std::abs(deltaEtaSuperClusterTrackAtVtx)            < 0.009"+
            "&& std::abs(deltaPhiSuperClusterTrackAtVtx)            < 0.10" +
            "&& sigmaIetaIeta                                  < 0.03" +
            "&& hcalOverEcal                                   < 0.10" +
            "&& std::abs(1./superCluster.energy - 1./p)             < 0.05))" 
            ),
        filter = cms.bool(False)
        )

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets as dummy
kt6PFJetsForRhoComputationVoronoiMet = dummy.clone(
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

from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits

hpsPFTauDiscriminationAgainstMuon2 = cms.EDProducer("PFRecoTauDiscriminationAgainstMuon2",
                                                         PFTauProducer = cms.InputTag('hpsPFTauProducer'),
                                                         Prediscriminants = requireDecayMode.clone(),
                                                         discriminatorOption = cms.string('loose'), # available options are: 'loose', 'medium', 'tight'
                                                         HoPMin = cms.double(0.2)
                                                     )


hpsPFTauDiscriminationByMVAIsolation = cms.EDProducer(
    "PFRecoTauDiscriminationByMVAIsolation",
            PFTauProducer = cms.InputTag('hpsPFTauProducer'),
            rhoProducer = cms.InputTag('kt6PFJetsForRhoComputationVoronoiMet','rho'),
            Prediscriminants = requireDecayMode.clone(),
            gbrfFilePath = cms.FileInPath('RecoTauTag/RecoTau/data/gbrfTauIso_v2.root'),
            returnMVA = cms.bool(False),
            mvaMin = cms.double(0.8),
            )

isotaus = cms.EDFilter(
    "PFTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    BooleanOperator = cms.string("and"),
    discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),       selectionCut=cms.double(0.5)),
    #cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByMVAIsolation"),           selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),           selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseElectronRejection"), selectionCut=cms.double(0.5)),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationAgainstMuon2"),             selectionCut=cms.double(0.5)) 
    ),
    cut = cms.string("std::abs(eta) < 2.3 && pt > 19.0 "),
    filter = cms.bool(False)
    )

isomuonseq     = cms.Sequence(isomuons)
isoelectronseq = cms.Sequence(isoelectrons)
isotauseq      = cms.Sequence(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits*
     #kt6PFJetsForRhoComputationVoronoiMet*
     #hpsPFTauDiscriminationByMVAIsolation*
     hpsPFTauDiscriminationAgainstMuon2*
     isotaus
    )

leptonSelection = cms.PSet(
    SelectEvents = cms.PSet(
    SelectEvents = cms.vstring(
    'isomuonseq',
    'isoelectronseq',
    'isotauseq')
    )
    )
