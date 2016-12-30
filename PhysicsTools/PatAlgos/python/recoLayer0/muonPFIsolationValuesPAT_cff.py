import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.muonPFIsolationValuesPFBRECO_cff import *

muPFIsoValueCharged03PAT = muPFIsoValueCharged03PFBRECO.clone()
muPFIsoValueCharged03PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFMeanDRIsoValueCharged03PAT = muPFMeanDRIsoValueCharged03PFBRECO.clone()
muPFMeanDRIsoValueCharged03PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFSumDRIsoValueCharged03PAT = muPFSumDRIsoValueCharged03PFBRECO.clone()
muPFSumDRIsoValueCharged03PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFIsoValueChargedAll03PAT = muPFIsoValueChargedAll03PFBRECO.clone()
muPFIsoValueChargedAll03PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFMeanDRIsoValueChargedAll03PAT = muPFMeanDRIsoValueChargedAll03PFBRECO.clone()
muPFMeanDRIsoValueChargedAll03PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFSumDRIsoValueChargedAll03PAT = muPFSumDRIsoValueChargedAll03PFBRECO.clone()
muPFSumDRIsoValueChargedAll03PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFIsoValueGamma03PAT = muPFIsoValueGamma03PFBRECO.clone()
muPFIsoValueGamma03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFMeanDRIsoValueGamma03PAT = muPFMeanDRIsoValueGamma03PFBRECO.clone()
muPFMeanDRIsoValueGamma03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFSumDRIsoValueGamma03PAT = muPFSumDRIsoValueGamma03PFBRECO.clone()
muPFSumDRIsoValueGamma03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFIsoValueNeutral03PAT = muPFIsoValueNeutral03PFBRECO.clone()
muPFIsoValueNeutral03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFMeanDRIsoValueNeutral03PAT = muPFMeanDRIsoValueNeutral03PFBRECO.clone()
muPFMeanDRIsoValueNeutral03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFSumDRIsoValueNeutral03PAT = muPFSumDRIsoValueNeutral03PFBRECO.clone()
muPFSumDRIsoValueNeutral03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFIsoValueGammaHighThreshold03PAT = muPFIsoValueGammaHighThreshold03PFBRECO.clone()
muPFIsoValueGammaHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFMeanDRIsoValueGammaHighThreshold03PAT = muPFMeanDRIsoValueGammaHighThreshold03PFBRECO.clone()
muPFMeanDRIsoValueGammaHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFSumDRIsoValueGammaHighThreshold03PAT = muPFSumDRIsoValueGammaHighThreshold03PFBRECO.clone()
muPFSumDRIsoValueGammaHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFIsoValueNeutralHighThreshold03PAT = muPFIsoValueNeutralHighThreshold03PFBRECO.clone()
muPFIsoValueNeutralHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFMeanDRIsoValueNeutralHighThreshold03PAT = muPFMeanDRIsoValueNeutralHighThreshold03PFBRECO.clone()
muPFMeanDRIsoValueNeutralHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFSumDRIsoValueNeutralHighThreshold03PAT = muPFSumDRIsoValueNeutralHighThreshold03PFBRECO.clone()
muPFSumDRIsoValueNeutralHighThreshold03PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFIsoValuePU03PAT = muPFIsoValuePU03PFBRECO.clone()
muPFIsoValuePU03PAT.deposits[0].src = 'muPFIsoDepositPUPAT'
muPFMeanDRIsoValuePU03PAT = muPFMeanDRIsoValuePU03PFBRECO.clone()
muPFMeanDRIsoValuePU03PAT.deposits[0].src = 'muPFIsoDepositPUPAT'
muPFSumDRIsoValuePU03PAT = muPFSumDRIsoValuePU03PFBRECO.clone()
muPFSumDRIsoValuePU03PAT.deposits[0].src = 'muPFIsoDepositPUPAT'
##############################
muPFIsoValueCharged04PAT = muPFIsoValueCharged04PFBRECO.clone()
muPFIsoValueCharged04PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFMeanDRIsoValueCharged04PAT = muPFMeanDRIsoValueCharged04PFBRECO.clone()
muPFMeanDRIsoValueCharged04PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFSumDRIsoValueCharged04PAT = muPFSumDRIsoValueCharged04PFBRECO.clone()
muPFSumDRIsoValueCharged04PAT.deposits[0].src = 'muPFIsoDepositChargedPAT'
muPFIsoValueChargedAll04PAT = muPFIsoValueChargedAll04PFBRECO.clone()
muPFIsoValueChargedAll04PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFMeanDRIsoValueChargedAll04PAT = muPFMeanDRIsoValueChargedAll04PFBRECO.clone()
muPFMeanDRIsoValueChargedAll04PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFSumDRIsoValueChargedAll04PAT = muPFSumDRIsoValueChargedAll04PFBRECO.clone()
muPFSumDRIsoValueChargedAll04PAT.deposits[0].src = 'muPFIsoDepositChargedAllPAT'
muPFIsoValueGamma04PAT = muPFIsoValueGamma04PFBRECO.clone()
muPFIsoValueGamma04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFMeanDRIsoValueGamma04PAT = muPFMeanDRIsoValueGamma04PFBRECO.clone()
muPFMeanDRIsoValueGamma04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFSumDRIsoValueGamma04PAT = muPFSumDRIsoValueGamma04PFBRECO.clone()
muPFSumDRIsoValueGamma04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFIsoValueNeutral04PAT = muPFIsoValueNeutral04PFBRECO.clone()
muPFIsoValueNeutral04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFMeanDRIsoValueNeutral04PAT = muPFMeanDRIsoValueNeutral04PFBRECO.clone()
muPFMeanDRIsoValueNeutral04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFSumDRIsoValueNeutral04PAT = muPFSumDRIsoValueNeutral04PFBRECO.clone()
muPFSumDRIsoValueNeutral04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFIsoValueGammaHighThreshold04PAT = muPFIsoValueGammaHighThreshold04PFBRECO.clone()
muPFIsoValueGammaHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFMeanDRIsoValueGammaHighThreshold04PAT = muPFMeanDRIsoValueGammaHighThreshold04PFBRECO.clone()
muPFMeanDRIsoValueGammaHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFSumDRIsoValueGammaHighThreshold04PAT = muPFSumDRIsoValueGammaHighThreshold04PFBRECO.clone()
muPFSumDRIsoValueGammaHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositGammaPAT'
muPFIsoValueNeutralHighThreshold04PAT = muPFIsoValueNeutralHighThreshold04PFBRECO.clone()
muPFIsoValueNeutralHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFMeanDRIsoValueNeutralHighThreshold04PAT = muPFMeanDRIsoValueNeutralHighThreshold04PFBRECO.clone()
muPFMeanDRIsoValueNeutralHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFSumDRIsoValueNeutralHighThreshold04PAT = muPFSumDRIsoValueNeutralHighThreshold04PFBRECO.clone()
muPFSumDRIsoValueNeutralHighThreshold04PAT.deposits[0].src = 'muPFIsoDepositNeutralPAT'
muPFIsoValuePU04PAT = muPFIsoValuePU04PFBRECO.clone()
muPFIsoValuePU04PAT.deposits[0].src = 'muPFIsoDepositPUPAT'
muPFMeanDRIsoValuePU04PAT = muPFMeanDRIsoValuePU04PFBRECO.clone()
muPFMeanDRIsoValuePU04PAT.deposits[0].src = 'muPFIsoDepositPUPAT'
muPFSumDRIsoValuePU04PAT = muPFSumDRIsoValuePU04PFBRECO.clone()
muPFSumDRIsoValuePU04PAT.deposits[0].src = 'muPFIsoDepositPUPAT'

muonPFIsolationValuesPATSequence = (
    muPFIsoValueCharged03PAT+
    muPFMeanDRIsoValueCharged03PAT+
    muPFSumDRIsoValueCharged03PAT+
    muPFIsoValueChargedAll03PAT+
    muPFMeanDRIsoValueChargedAll03PAT+
    muPFSumDRIsoValueChargedAll03PAT+
    muPFIsoValueGamma03PAT+
    muPFMeanDRIsoValueGamma03PAT+
    muPFSumDRIsoValueGamma03PAT+
    muPFIsoValueNeutral03PAT+
    muPFMeanDRIsoValueNeutral03PAT+
    muPFSumDRIsoValueNeutral03PAT+
    muPFIsoValueGammaHighThreshold03PAT+
    muPFMeanDRIsoValueGammaHighThreshold03PAT+
    muPFSumDRIsoValueGammaHighThreshold03PAT+
    muPFIsoValueNeutralHighThreshold03PAT+
    muPFMeanDRIsoValueNeutralHighThreshold03PAT+
    muPFSumDRIsoValueNeutralHighThreshold03PAT+
    muPFIsoValuePU03PAT+
    muPFMeanDRIsoValuePU03PAT+
    muPFSumDRIsoValuePU03PAT+
    ##############################
    muPFIsoValueCharged04PAT+
    muPFMeanDRIsoValueCharged04PAT+
    muPFSumDRIsoValueCharged04PAT+
    muPFIsoValueChargedAll04PAT+
    muPFMeanDRIsoValueChargedAll04PAT+
    muPFSumDRIsoValueChargedAll04PAT+
    muPFIsoValueGamma04PAT+
    muPFMeanDRIsoValueGamma04PAT+
    muPFSumDRIsoValueGamma04PAT+
    muPFIsoValueNeutral04PAT+
    muPFMeanDRIsoValueNeutral04PAT+
    muPFSumDRIsoValueNeutral04PAT+
    muPFIsoValueGammaHighThreshold04PAT+
    muPFMeanDRIsoValueGammaHighThreshold04PAT+
    muPFSumDRIsoValueGammaHighThreshold04PAT+
    muPFIsoValueNeutralHighThreshold04PAT+
    muPFMeanDRIsoValueNeutralHighThreshold04PAT+
    muPFSumDRIsoValueNeutralHighThreshold04PAT+
    muPFIsoValuePU04PAT+
    muPFMeanDRIsoValuePU04PAT+
    muPFSumDRIsoValuePU04PAT
    )
