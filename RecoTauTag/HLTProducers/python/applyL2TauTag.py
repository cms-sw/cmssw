import FWCore.ParameterSet.Config as cms

from RecoTauTag.HLTProducers.l2TauNNProducer_cfi import *
from RecoTauTag.HLTProducers.l2TauTagFilter_cfi import *

def insertL2TauSequence(process, path, ref_module):
    ref_idx = path.index(ref_module)
    path.insert(ref_idx + 1, process.hltL2TauTagNNSequence)
    path.insert(ref_idx + 2, process.hltL2DoubleTauTagNNFilter)
    path.insert(ref_idx + 3, process.HLTGlobalPFTauHPSSequence)


def update(process):
    thWp = {
            'Tight': 0.180858813224404,
            'Medium': 0.12267940863785043,
            'Loose': 0.08411243185219064,
    }

    working_point = "Tight"
    graphPath = 'RecoTauTag/TrainingFiles/data/L2TauNNTag/L2TauTag_Run3v1.pb'

    normalizationDict = 'RecoTauTag/TrainingFiles/data/L2TauNNTag/NormalizationDict.json'

    process.hltL2TauTagNNProducer = l2TauNNProducer.clone(
        debugLevel = 0,
        L1Taus = [
            cms.PSet(
                L1CollectionName = cms.string('DoubleTau'),
                L1TauTrigger = cms.InputTag('hltL1sDoubleTauBigOR'),
            ),
        ],
        hbheInput = "hltHbhereco",
        hoInput = "hltHoreco",
        ebInput = "hltEcalRecHit:EcalRecHitsEB",
        eeInput = "hltEcalRecHit:EcalRecHitsEE",
        pataVertices = "hltPixelVerticesSoA",
        pataTracks = "hltPixelTracksSoA",
        BeamSpot = "hltOnlineBeamSpot",
        maxVtx = 100,
        fractionSumPt2 = 0.3,
        minSumPt2 = 0.,
        track_pt_min = 1.,
        track_pt_max = 20.,
        track_chi2_max = 20.,
        graphPath = graphPath,
        normalizationDict = normalizationDict
    )
    process.hltL2DoubleTauTagNNFilter = l2TauTagFilter.clone(
        nExpected = 2,
        L1TauSrc = 'hltL1sDoubleTauBigOR',
        L2Outcomes = 'hltL2TauTagNNProducer:DoubleTau',
        DiscrWP = thWp[working_point],
        l1TauPtThreshold = 250,
    )
    # L2 updated Sequence
    process.hltL2TauTagNNSequence = cms.Sequence(process.HLTDoCaloSequence + process.hltL1sDoubleTauBigOR + process.hltL2TauTagNNProducer)


    # Regional -> Global customization
    process.hltHpsPFTauTrackPt1DiscriminatorReg.PFTauProducer = "hltHpsPFTauProducer"
    process.hltHpsDoublePFTau35Reg.inputTag = "hltHpsPFTauProducer"
    process.hltHpsSelectedPFTausTrackPt1Reg.src = "hltHpsPFTauProducer"
    process.hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminatorReg.PFTauProducer = "hltHpsPFTauProducer"
    process.hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminatorReg.particleFlowSrc = "hltParticleFlow"
    process.hltHpsPFTauMediumRelativeChargedIsolationDiscriminatorReg.PFTauProducer = "hltHpsPFTauProducer"
    process.hltHpsPFTauMediumRelativeChargedIsolationDiscriminatorReg.particleFlowSrc = "hltParticleFlow"
    process.hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminatorReg.PFTauProducer = "hltHpsPFTauProducer"
    process.hltHpsSelectedPFTausTrackPt1MediumChargedIsolationReg.src = "hltHpsPFTauProducer"

    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.HLTL2TauJetsL1TauSeededSequence)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltDoubleL2Tau26eta2p2)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.HLTL2p5IsoTauL1TauSeededSequence)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltDoubleL2IsoTau26eta2p2 )
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.HLTRegionalPFTauHPSSequence )

    insertL2TauSequence(process, process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4, process.hltPreDoubleMediumChargedIsoPFTauHPS35Trk1eta2p1Reg)


    old_diTau_paths = ['HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1_v1', 'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1_v1','HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1_v1','HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1_v4','HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS30_Trk1_eta2p1_Reg_CrossL1_v1','HLT_DoubleMediumChargedIsoPFTauHPS30_L1MaxMass_Trk1_eta2p1_Reg_v1','HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v1','HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_v1','HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_v1','HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1_Reg_v1','HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_eta2p1_Reg_v1','HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg_v1','HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg_v1']
    for path in old_diTau_paths:
        if path in process.__dict__:
            process.schedule.remove(getattr(process, path))


    return process
