import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

BTagAndProbe_TnP = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BTV/TnP/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "jet_eta 'BTag and Probe jet eta efficiency; jet #eta; efficiency' jet_eta_numerator jet_eta_denominator",
        "jet_pt 'BTag and Probe jet pt efficiency; jet pt; efficiency' jet_pt_numerator jet_pt_denominator",
        "vertexMass 'BTag and Probe vertexMass efficiency; vertexMass; efficiency' vertexMass_numerator vertexMass_denominator",
        "jetNSecondaryVertices 'BTag and Probe jetNSecondaryVertices efficiency; jetNSecondaryVertices; efficiency' jetNSecondaryVertices_numerator jetNSecondaryVertices_denominator",
        "trackSumJetEtRatio 'BTag and Probe trackSumJetEtRatio efficiency; jet trackSumJetEtRatio; efficiency' trackSumJetEtRatio_numerator trackSumJetEtRatio_denominator",
        "trackSip2dValAboveCharm 'BTag and Probe trackSip2dValAboveCharm efficiency; trackSip2dValAboveCharm; efficiency' trackSip2dValAboveCharm_numerator trackSip2dValAboveCharm_denominator",
        "trackSip2dSigAboveCharm 'BTag and Probe trackSip2dSigAboveCharm efficiency; trackSip2dValAboveCharm; efficiency' trackSip2dValAboveCharm_numerator trackSip2dValAboveCharm_denominator",
        "trackSip3dValAboveCharm 'BTag and Probe trackSip3dValAboveCharm efficiency; trackSip3dValAboveCharm; efficiency' trackSip3dValAboveCharm_numerator trackSip3dValAboveCharm_denominator",
        "trackSip3dSigAboveCharm 'BTag and Probe trackSip3dSigAboveCharm efficiency; trackSip3dSigAboveCharm; efficiency' trackSip3dSigAboveCharm_numerator trackSip3dSigAboveCharm_denominator",
        "jetNSelectedTracks 'BTag and Probe jetNSelectedTracks efficiency; jetNSelectedTracks; efficiency' jetNSelectedTracks_numerator jetNSelectedTracks_denominator",
        "jetNTracksEtaRel 'BTag and Probe jetNTracksEtaRel efficiency; jetNTracksEtaRel; efficiency' jetNTracksEtaRel_numerator jetNTracksEtaRel_denominator",
        "vertexCategory 'BTag and Probe vertexCategory efficiency; vertexCategory; efficiency' vertexCategory_numerator vertexCategory_denominator",
        "trackSumJetDeltaR 'BTag and Probe trackSumJetDeltaR efficiency; trackSumJetDeltaR; efficiency' trackSumJetDeltaR_numerator trackSumJetDeltaR_denominator",
        "trackJetDistVal 'BTag and Probe trackJetDistVal efficiency; trackJetDistVal; efficiency' trackJetDistVal_numerator trackJetDistVal_denominator",
        "trackPtRel 'BTag and Probe trackPtRel efficiency; trackPtRel; efficiency' trackPtRel_numerator trackPtRel_denominator",
        "trackDeltaR 'BTag and Probe trackDeltaR efficiency; trackDeltaR; efficiency' trackDeltaR_numerator trackDeltaR_denominator",
        "trackPtRatio 'BTag and Probe trackPtRatio efficiency; jet trackPtRatio; efficiency' trackPtRatio_numerator trackPtRatio_denominator",
        "trackSip2dSig 'BTag and Probe trackSip2dSig efficiency; trackSip2dSig; efficiency' trackSip2dSig_numerator trackSip2dSig_denominator",
        "trackSip3dSig 'BTag and Probe trackSip3dSig efficiency; trackSip3dSig; efficiency' trackSip3dSig_numerator trackSip3dSig_denominator",
        "trackDecayLenVal 'BTag and Probe trackDecayLenVal efficiency; trackDecayLenVal; efficiency' trackDecayLenVal_numerator trackDecayLenVal_denominator",
        "trackEtaRel 'BTag and Probe trackEtaRel efficiency; trackEtaRel; efficiency' trackEtaRel_numerator trackEtaRel_denominator",
        "vertexEnergyRatio 'BTag and Probe vertexEnergyRatio efficiency; vertexEnergyRatio; efficiency' vertexEnergyRatio_numerator vertexEnergyRatio_denominator",
        "vertexJetDeltaR 'BTag and Probe vertexJetDeltaR efficiency; jetNTracksEtaRel; efficiency' vertexJetDeltaR_numerator vertexJetDeltaR_denominator",
        "vertexNTracks 'BTag and Probe vertexNTracks efficiency; vertexNTracks; efficiency' vertexNTracks_numerator vertexNTracks_denominator",
        "flightDistance2dVal 'BTag and Probe flightDistance2dVal efficiency; flightDistance2dVal; efficiency' flightDistance2dVal_numerator flightDistance2dVal_denominator",
        "flightDistance2dSig 'BTag and Probe flightDistance2dSig efficiency; flightDistance2dSig; efficiency' flightDistance2dSig_numerator flightDistance2dSig_denominator",
        "flightDistance3dVal 'BTag and Probe flightDistance3dVal efficiency; flightDistance3dVal; efficiency' flightDistance3dVal_numerator flightDistance3dVal_denominator",
        "flightDistance3dSig 'BTag and Probe flightDistance3dSig efficiency; flightDistance3dSig; efficiency' flightDistance3dSig_numerator flightDistance3dSig_denominator",
    ),
)

BTagAndProbeClient = cms.Sequence(
    BTagAndProbe_TnP
)
