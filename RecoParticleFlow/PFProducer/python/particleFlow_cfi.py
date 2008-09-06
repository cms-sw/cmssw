import FWCore.ParameterSet.Config as cms

particleFlow = cms.EDProducer("PFProducer",
    verbose = cms.untracked.bool(False),
    pf_calib_HCAL_offset = cms.double(1.73),
    pf_nsigma_ECAL = cms.double(3.0),
    pf_calib_ECAL_HCAL_hslope = cms.double(1.06),
    usePFElectrons = cms.bool(True),
    final_chi2cut_bremps = cms.double(25.0),
    pf_mergedPhotons_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_MLP.weights.txt'),
    pf_calib_ECAL_offset = cms.double(0.0),
    pf_electron_mvaCut = cms.double(-0.4),
    pf_calib_ECAL_HCAL_eslope = cms.double(1.05),
    pf_mergedPhotons_mvaCut = cms.double(0.5),
    final_chi2cut_gsfhcal = cms.double(100.0),
    pf_calib_ECAL_HCAL_offset = cms.double(6.11),
    final_chi2cut_gsfps = cms.double(100.0),
    pf_nsigma_HCAL = cms.double(1.0),
    blocks = cms.InputTag("particleFlowBlock"),
    pf_calib_ECAL_slope = cms.double(1.0),
    pf_clusterRecovery = cms.bool(False),
    final_chi2cut_bremecal = cms.double(25.0),
    pf_calib_HCAL_damping = cms.double(2.49),
    final_chi2cut_gsfecal = cms.double(900.0),
    pf_electronID_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_finalID_hzz-pions.txt'),
    pf_calib_HCAL_slope = cms.double(2.17),
    pf_mergedPhotons_PSCut = cms.double(0.001),
    algoType = cms.uint32(0),
    final_chi2cut_bremhcal = cms.double(25.0),
    debug = cms.untracked.bool(False)
)



