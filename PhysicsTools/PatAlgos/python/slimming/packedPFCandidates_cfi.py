import FWCore.ParameterSet.Config as cms

packedPFCandidates = cms.EDProducer("PATPackedCandidateProducer",
    inputCollection = cms.InputTag("particleFlow"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    originalTracks = cms.InputTag("generalTracks"),
    vertexAssociator = cms.InputTag("primaryVertexAssociation","original"),
    PuppiSrc = cms.InputTag("puppi"),
    PuppiNoLepSrc = cms.InputTag("puppiNoLep"),    
    chargedHadronIsolation = cms.InputTag("chargedHadronPFTrackIsolation"),
    minPtForChargedHadronProperties = cms.double(3.0),
    secondaryVerticesForWhiteList = cms.VInputTag(
      cms.InputTag("inclusiveCandidateSecondaryVertices"),
      cms.InputTag("inclusiveCandidateSecondaryVerticesCvsL"),
      cms.InputTag("generalV0Candidates","Kshort"),
      cms.InputTag("generalV0Candidates","Lambda"),
      ),      
    minPtForTrackProperties = cms.double(0.95),
    covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1   
#    covariancePackingSchemas = cms.vint32(1,257,513,769,0),  # a cheaper schema in kb/ev 
    covariancePackingSchemas = cms.vint32(8,264,520,776,0),   # more accurate schema +0.6kb/ev
    pfCandidateTypesForHcalDepth = cms.vint32(),
    storeHcalDepthEndcapOnly = cms.bool(False), # switch to store info only for endcap 
    storeTiming = cms.bool(False),
    timeMap = cms.InputTag(""),
    timeMapErr = cms.InputTag("")
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(packedPFCandidates, covarianceVersion =1 )

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(packedPFCandidates, chargedHadronIsolation = "" )

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(packedPFCandidates,
    pfCandidateTypesForHcalDepth = [130,11,22,211,13],  # PF cand types for adding Hcal depth energy frac information
                        # (130: neutral h, 11: ele, 22: photon, 211: charged h, 13: mu) # excluding e.g. 1:h_HF, 2:egamma_HF
    storeHcalDepthEndcapOnly = True
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(packedPFCandidates,
    pfCandidateTypesForHcalDepth = [], # For now, no PF cand type is considered for addition of Hcal depth energy frac 
    storeHcalDepthEndcapOnly = False
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(packedPFCandidates, storeTiming = cms.bool(True))

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(packedPFCandidates, PuppiSrc = "", PuppiNoLepSrc = "")
from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toModify(packedPFCandidates,
                                    inputCollection = "cleanedParticleFlow",
                                    chargedHadronIsolation = ""
                                )
