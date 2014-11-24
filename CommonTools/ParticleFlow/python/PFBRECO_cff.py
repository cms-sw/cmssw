import FWCore.ParameterSet.Config as cms

# getting the ptrs
from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs

from CommonTools.ParticleFlow.pfPileUp_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *
pfPileUpIsoPFBRECO = pfPileUp.clone( PFCandidates = 'particleFlowPtrs' )
pfNoPileUpIsoPFBRECO = pfNoPileUp.clone( topCollection = 'pfPileUpIsoPFBRECO',
                                         bottomCollection = 'particleFlowPtrs')
pfNoPileUpIsoPFBRECOSequence = cms.Sequence(
    pfPileUpIsoPFBRECO +
    pfNoPileUpIsoPFBRECO
    )

from CommonTools.ParticleFlow.pfNoPileUpJME_cff import *

pfPileUpPFBRECO = pfPileUp.clone( PFCandidates = 'particleFlowPtrs' )
pfNoPileUpPFBRECO = pfNoPileUp.clone( topCollection = 'pfPileUpPFBRECO',
                                      bottomCollection = 'particleFlowPtrs')
pfNoPileUpPFBRECOSequence = cms.Sequence(
    pfPileUpPFBRECO +
    pfNoPileUpPFBRECO
    )

from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadrons_cfi import *
pfAllNeutralHadronsPFBRECO = pfAllNeutralHadrons.clone( src = 'pfNoPileUpIsoPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedHadrons_cfi import *
pfAllChargedHadronsPFBRECO = pfAllChargedHadrons.clone( src = 'pfNoPileUpIsoPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllPhotons_cfi import *
pfAllPhotonsPFBRECO = pfAllPhotons.clone( src = 'pfNoPileUpIsoPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi import *
pfAllMuonsPFBRECO = pfAllMuons.clone( src = 'pfNoPileUpPFBRECO' )
pfAllMuonsClonesPFBRECO = pfAllMuonsClones.clone( src = 'pfAllMuonsPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi import *
pfAllElectronsPFBRECO = pfAllElectrons.clone( src = 'pfNoMuonPFBRECO' )
pfAllElectronsClonesPFBRECO = pfAllElectronsClones.clone( src = 'pfAllElectronsPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedParticles_cfi import *
pfAllChargedParticlesPFBRECO = pfAllChargedParticles.clone( src = 'pfNoPileUpIsoPFBRECO' )
from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadronsAndPhotons_cfi import *
pfAllNeutralHadronsAndPhotonsPFBRECO = pfAllNeutralHadronsAndPhotons.clone( src = 'pfNoPileUpIsoPFBRECO' )
pfPileUpAllChargedParticlesPFBRECO = pfAllChargedParticles.clone( src = 'pfPileUpIsoPFBRECO' )
pfSortByTypePFBRECOSequence = cms.Sequence(
    pfAllNeutralHadronsPFBRECO+
    pfAllChargedHadronsPFBRECO+
    pfAllPhotonsPFBRECO+
    # charged hadrons + electrons + muons
    pfAllChargedParticlesPFBRECO+
    # same, but from pile up
    pfPileUpAllChargedParticlesPFBRECO+
    pfAllNeutralHadronsAndPhotonsPFBRECO
#    +
#    pfAllElectronsPFBRECO+
#    pfAllMuonsPFBRECO
    )

pfParticleSelectionPFBRECOSequence = cms.Sequence(
    pfNoPileUpIsoPFBRECOSequence +
    # In principle JME sequence should go here, but this is used in RECO
    # in addition to here, and is used in the "first-step" PF process
    # so needs to go later.
    #pfNoPileUpJMESequence +
    pfNoPileUpPFBRECOSequence +
    pfSortByTypePFBRECOSequence
    )

from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedPhotons_cfi import *
pfSelectedPhotonsPFBRECO = pfSelectedPhotons.clone( src = 'pfAllPhotonsPFBRECO' )
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolationPFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedPhotons_cfi import *
pfIsolatedPhotonsPFBRECO = pfIsolatedPhotons.clone( src = 'pfSelectedPhotonsPFBRECO',
                                                    isolationValueMapsCharged = cms.VInputTag( cms.InputTag("phPFIsoValueCharged04PFIdPFBRECO") ),
                                                    isolationValueMapsNeutral = cms.VInputTag( cms.InputTag("phPFIsoValueNeutral04PFIdPFBRECO"),
                                                                                               cms.InputTag("phPFIsoValueGamma04PFIdPFBRECO") ),
                                                    deltaBetaIsolationValueMap = 'phPFIsoValuePU04PFIdPFBRECO' )
pfPhotonPFBRECOSequence = cms.Sequence(
    pfSelectedPhotonsPFBRECO +
    pfPhotonIsolationPFBRECOSequence +
    # selecting isolated photons:
    pfIsolatedPhotonsPFBRECO
    )

from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsFromVertex_cfi import *
pfMuonsFromVertexPFBRECO = pfMuonsFromVertex.clone( src = 'pfAllMuonsPFBRECO' )
from CommonTools.ParticleFlow.Isolation.pfIsolatedMuons_cfi import *
pfIsolatedMuonsPFBRECO = pfIsolatedMuons.clone( src = 'pfMuonsFromVertexPFBRECO' )
pfMuonsPFBRECO = pfIsolatedMuonsPFBRECO.clone(cut = cms.string("pt > 5 & muonRef.isAvailable()"))
pfMuonPFBRECOSequence = cms.Sequence(
    pfAllMuonsPFBRECO +
    pfMuonsFromVertexPFBRECO +
    pfIsolatedMuonsPFBRECO+
    pfMuonsPFBRECO
    )

from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import *
pfElectronsFromVertexPFBRECO = pfElectronsFromVertex.clone( src = 'pfAllElectronsPFBRECO' )
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *
pfIsolatedElectronsPFBRECO = pfIsolatedElectrons.clone( src = 'pfElectronsFromVertexPFBRECO' )
pfElectronsPFBRECO = pfIsolatedElectronsPFBRECO.clone( cut = cms.string(" pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.hitPattern().numberOfLostHits('MISSING_INNER_HITS')<2"))
pfElectronPFBRECOSequence = cms.Sequence(
    pfAllElectronsPFBRECO +
    pfElectronsFromVertexPFBRECO +
    pfIsolatedElectronsPFBRECO +
    pfElectronsPFBRECO
    )

from CommonTools.ParticleFlow.Tools.jetTools import jetAlgo
pfJetsPFBRECO = jetAlgo('AK4')
pfJetsPFBRECO.src = 'pfNoElectronJMEPFBRECO'
pfJetsPtrsPFBRECO = cms.EDProducer("PFJetFwdPtrProducer",
                                   src=cms.InputTag("pfJetsPFBRECO")
                                   )
pfJetPFBRECOSequence = cms.Sequence(
    pfJetsPFBRECO +
    pfJetsPtrsPFBRECO
    )

from CommonTools.ParticleFlow.pfTaus_cff import *

from CommonTools.ParticleFlow.pfMET_cfi import *
pfMETPFBRECO = pfMET.clone( jets = 'pfJetsPFBRECO' )

##delta beta weighting
#from CommonTools.ParticleFlow.deltaBetaWeights_cfi import *
#pfWeightedPhotonsPFBRECO = pfWeightedPhotons.clone( src = 'pfAllPhotonsPFBRECO',
                                                    #chargedFromPV = 'pfAllChargedParticlesPFBRECO',
                                                    #chargedFromPU = 'pfPileUpAllChargedParticlesPFBRECO' )
#pfWeightedNeutralHadronsPFBRECO = pfWeightedNeutralHadrons.clone( src = 'pfAllNeutralHadronsPFBRECO',
                                                                  #chargedFromPV = 'pfAllChargedParticlesPFBRECO',
                                                                  #chargedFromPU = 'pfPileUpAllChargedParticlesPFBRECO' )
#pfDeltaBetaWeightingPFBRECOSequence = cms.Sequence(pfWeightedPhotonsPFBRECO+pfWeightedNeutralHadronsPFBRECO)

# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import *
pfNoMuonPFBRECO = pfNoMuon.clone( topCollection = 'pfIsolatedMuonsPFBRECO',
                                  bottomCollection = 'pfNoPileUpPFBRECO' )
pfNoMuonJMEPFBRECO = pfNoMuonJME.clone( topCollection = 'pfIsolatedMuonsPFBRECO' )
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import *
pfNoElectronPFBRECO = pfNoElectron.clone( topCollection = 'pfIsolatedElectronsPFBRECO',
                                          bottomCollection = 'pfNoMuonPFBRECO' )
pfNoElectronJMEPFBRECO = pfNoElectronJME.clone( topCollection = 'pfIsolatedElectronsPFBRECO',
                                                bottomCollection = 'pfNoMuonJMEPFBRECO' )
pfNoElectronJMEClonesPFBRECO = pfNoElectronJMEClones.clone( src = 'pfNoElectronJMEPFBRECO' )
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cff import *
pfNoJetPFBRECO = pfNoJet.clone( topCollection = 'pfJetsPtrsPFBRECO',
                                bottomCollection = 'pfNoElectronJMEPFBRECO' )
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cff import *
pfNoTauPFBRECO = pfNoTau.clone ( bottomCollection = 'pfJetsPtrsPFBRECO' )
pfNoTauClonesPFBRECO = pfNoTauClones.clone ( src = 'pfNoTauPFBRECO' )

# generator tools
from CommonTools.ParticleFlow.genForPF2PAT_cff import *

PFBRECO = cms.Sequence(
    particleFlowPtrs +
    pfParticleSelectionPFBRECOSequence +
    pfNoPileUpJMESequence +
#    pfDeltaBetaWeightingPFBRECOSequence +
    pfPhotonPFBRECOSequence +
    pfMuonPFBRECOSequence +
    pfNoMuonPFBRECO +
    pfNoMuonJMEPFBRECO +
    pfElectronPFBRECOSequence +
    pfNoElectronPFBRECO +
    pfNoElectronJMEPFBRECO +
    pfNoElectronJMEClonesPFBRECO+
    pfJetPFBRECOSequence +
    pfNoJetPFBRECO +
    pfTauSequence +
    pfNoTauPFBRECO +
    pfMETPFBRECO
    )
