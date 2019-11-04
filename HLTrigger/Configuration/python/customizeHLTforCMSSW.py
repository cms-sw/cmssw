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
        toDelete=['algoType',
                  'isolatedElectronID_mvaWeightFile',
                  'pf_electronID_mvaWeightFile',
                  'pf_electron_output_col',
                  'minTrackerHits',
                  'minPixelHits',
                  'dzPV']
        #
        # kill parameters that are moved to sub-psets
        # PFEGammaFiltersParameters
        toDelete.extend(['electron_iso_pt',
                         'electron_iso_mva_barrel',
                         'electron_iso_mva_endcap',
                         'electron_iso_combIso_barrel',
                         'electron_iso_combIso_endcap',
                         'electron_noniso_mvaCut',
                         'electron_missinghits',
                         'electron_ecalDrivenHademPreselCut',
                         'electron_maxElePtForOnlyMVAPresel',
                         'electron_protectionsForJetMET',
                         'electron_protectionsForBadHcal',
                         'photon_MinEt',
                         'photon_combIso',
                         'photon_HoE',
                         'photon_SigmaiEtaiEta_barrel',
                         'photon_SigmaiEtaiEta_endcap',
                         'photon_protectionsForJetMET',
                         'photon_protectionsForBadHcal'
                         ])
        # PFMuonAlgoParameters
        toDelete.extend(['maxDPtOPt',
                         'trackQuality',
                         'ptErrorScale',
                         'eventFractionForCleaning',
                         'minPtForPostCleaning',
                         'eventFactorForCosmics',
                         'metSignificanceForCleaning',
                         'metSignificanceForRejection',
                         'metFactorForCleaning',
                         'eventFractionForRejection',
                         'metFactorForRejection',
                         'metFactorForHighEta',
                         'ptFactorForHighEta',
                         'metFactorForFakes',
                         'minMomentumForPunchThrough',
                         'minEnergyForPunchThrough',
                         'punchThroughFactor',
                         'punchThroughMETFactor',
                         'cosmicRejectionDistance'])
        # Post HF cleaning
        toDelete.extend(['minHFCleaningPt',
                         'maxSignificance',
                         'minSignificance',
                         'minSignificanceReduction',
                         'maxDeltaPhiPt',
                         'minDeltaMet'])
        #
        # Actually kill them
        for att in toDelete:
            if (hasattr(producer, att)): delattr(producer, att)
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
