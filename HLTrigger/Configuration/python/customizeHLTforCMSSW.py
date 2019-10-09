import FWCore.ParameterSet.Config as cms

# helper fuctions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

#
# PF config cleanup
def customiseFor28110(process):

    for producer in producers_by_type(process, "PFProducer"):
        #
        # kill cleaned-up parameters
        if hasattr(producer, "iCfgCandConnector"):
            pset = getattr(producer, "iCfgCandConnector")
            delattr(pset,'bCalibSecondary')
        if hasattr(producer, "algoType"): delattr(producer, "algoType")
        if hasattr(producer, "isolatedElectronID_mvaWeightFile"): delattr(producer, "isolatedElectronID_mvaWeightFile")
        if hasattr(producer, "pf_electronID_mvaWeightFile"): delattr(producer, "pf_electronID_mvaWeightFile")
        if hasattr(producer, "pf_electron_output_col"): delattr(producer, "pf_electron_output_col")
        if hasattr(producer, "minTrackerHits"): delattr(producer, "minTrackerHits")
        if hasattr(producer, "minPixelHits"): delattr(producer, "minPixelHits")
        if hasattr(producer, "dzPV"): delattr(producer, "dzPV")
        #
        # kill parameters that are moved to sub-psets
        # PFEgammaFiltersParameters
        if hasattr(producer, "electron_iso_pt"): delattr(producer, "electron_iso_pt")
        if hasattr(producer, "electron_iso_mva_barrel"): delattr(producer, "electron_iso_mva_barrel")
        if hasattr(producer, "electron_iso_mva_endcap"): delattr(producer, "electron_iso_mva_endcap")
        if hasattr(producer, "electron_iso_combIso_barrel"): delattr(producer, "electron_iso_combIso_barrel")
        if hasattr(producer, "electron_iso_combIso_endcap"): delattr(producer, "electron_iso_combIso_endcap")
        if hasattr(producer, "electron_noniso_mvaCut"): delattr(producer, "electron_noniso_mvaCut")
        if hasattr(producer, "electron_missinghits"): delattr(producer, "electron_missinghits")
        if hasattr(producer, "electron_ecalDrivenHademPreselCut"): delattr(producer, "electron_ecalDrivenHademPreselCut")
        if hasattr(producer, "electron_maxElePtForOnlyMVAPresel"): delattr(producer, "electron_maxElePtForOnlyMVAPresel")
        if hasattr(producer, "electron_protectionsForJetMET"): delattr(producer, "electron_protectionsForJetMET")
        if hasattr(producer, "electron_protectionsForBadHcal"): delattr(producer, "electron_protectionsForBadHcal")
        if hasattr(producer, "photon_MinEt"): delattr(producer, "photon_MinEt")
        if hasattr(producer, "photon_combIso"): delattr(producer, "photon_combIso")
        if hasattr(producer, "photon_HoE"): delattr(producer, "photon_HoE")
        if hasattr(producer, "photon_SigmaiEtaiEta_barrel"): delattr(producer, "photon_SigmaiEtaiEta_barrel")
        if hasattr(producer, "photon_SigmaiEtaiEta_endcap"): delattr(producer, "photon_SigmaiEtaiEta_endcap")
        if hasattr(producer, "photon_protectionsForJetMET"): delattr(producer, "photon_protectionsForJetMET")
        if hasattr(producer, "photon_protectionsForBadHcal"): delattr(producer, "photon_protectionsForBadHcal")
        # PFMuonAlgoParameters
        if hasattr(producer, "maxDPtOPt"): delattr(producer, "maxDPtOPt")
        if hasattr(producer, "trackQuality"): delattr(producer, "trackQuality")
        if hasattr(producer, "ptErrorScale"): delattr(producer, "ptErrorScale")
        if hasattr(producer, "eventFractionForCleaning"): delattr(producer, "eventFractionForCleaning")
        if hasattr(producer, "minPtForPostCleaning"): delattr(producer, "minPtForPostCleaning")
        if hasattr(producer, "eventFactorForCosmics"): delattr(producer, "eventFactorForCosmics")
        if hasattr(producer, "metSignificanceForCleaning"): delattr(producer, "metSignificanceForCleaning")
        if hasattr(producer, "metSignificanceForRejection"): delattr(producer, "metSignificanceForRejection")
        if hasattr(producer, "metFactorForCleaning"): delattr(producer, "metFactorForCleaning")
        if hasattr(producer, "eventFractionForRejection"): delattr(producer, "eventFractionForRejection")
        if hasattr(producer, "metFactorForRejection"): delattr(producer, "metFactorForRejection")
        if hasattr(producer, "metFactorForHighEta"): delattr(producer, "metFactorForHighEta")
        if hasattr(producer, "ptFactorForHighEta"): delattr(producer, "ptFactorForHighEta")
        if hasattr(producer, "metFactorForFakes"): delattr(producer, "metFactorForFakes")
        if hasattr(producer, "minMomentumForPunchThrough"): delattr(producer, "minMomentumForPunchThrough")
        if hasattr(producer, "minEnergyForPunchThrough"): delattr(producer, "minEnergyForPunchThrough")
        if hasattr(producer, "punchThroughFactor"): delattr(producer, "punchThroughFactor")
        if hasattr(producer, "punchThroughMETFactor"): delattr(producer, "punchThroughMETFactor")
        if hasattr(producer, "cosmicRejectionDistance"): delattr(producer, "cosmicRejectionDistance")
        # Post HF cleaning
        if hasattr(producer, "minHFCleaningPt"): delattr(producer, "minHFCleaningPt")
        if hasattr(producer, "maxSignificance"): delattr(producer, "maxSignificance")
        if hasattr(producer, "minSignificance"): delattr(producer, "minSignificance")
        if hasattr(producer, "minSignificanceReduction"): delattr(producer, "minSignificanceReduction")
        if hasattr(producer, "maxDeltaPhiPt"): delattr(producer, "maxDeltaPhiPt")
        if hasattr(producer, "minDeltaMet"): delattr(producer, "minDeltaMet")
        #
    return process

def customiseFor2017DtUnpacking(process):
    """Adapt the HLT to run the legacy DT unpacking
    for pre2018 data/MC workflows as the default"""

    if hasattr(process,'hltMuonDTDigis'):
        process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
            useStandardFEDid = cms.bool( True ),
            maxFEDid = cms.untracked.int32( 779 ),
            inputLabel = cms.InputTag( "rawDataCollector" ),
            minFEDid = cms.untracked.int32( 770 ),
            dataType = cms.string( "DDU" ),
            readOutParameters = cms.PSet(
                localDAQ = cms.untracked.bool( False ),
                debug = cms.untracked.bool( False ),
                rosParameters = cms.PSet(
                    localDAQ = cms.untracked.bool( False ),
                    debug = cms.untracked.bool( False ),
                    writeSC = cms.untracked.bool( True ),
                    readDDUIDfromDDU = cms.untracked.bool( True ),
                    readingDDU = cms.untracked.bool( True ),
                    performDataIntegrityMonitor = cms.untracked.bool( False )
                    ),
                performDataIntegrityMonitor = cms.untracked.bool( False )
                ),
            dqmOnly = cms.bool( False )
        )

    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customiseFor28110(process)

    return process
