## list of all available btagInfos
supportedBtagInfos = [
    'None'
    # legacy framework (supported with RECO/AOD but not MiniAOD)
  , 'impactParameterTagInfos'
  , 'secondaryVertexTagInfos'
  , 'secondaryVertexNegativeTagInfos'
  , 'softMuonTagInfos'
  , 'inclusiveSecondaryVertexFinderTagInfos'
  , 'inclusiveSecondaryVertexFinderNegativeTagInfos'
  , 'inclusiveSecondaryVertexFinderFilteredTagInfos'
  , 'inclusiveSecondaryVertexFinderFilteredNegativeTagInfos'
    # new candidate-based framework (supported with RECO/AOD/MiniAOD)
  , 'pfImpactParameterTagInfos'
  , 'pfImpactParameterAK8TagInfos'
  , 'pfImpactParameterCA15TagInfos'
  , 'pfSecondaryVertexTagInfos'
  , 'pfSecondaryVertexNegativeTagInfos'
  , 'pfInclusiveSecondaryVertexFinderTagInfos'
  , 'pfInclusiveSecondaryVertexFinderAK8TagInfos'
  , 'pfInclusiveSecondaryVertexFinderCA15TagInfos'
  , 'pfInclusiveSecondaryVertexFinderNegativeTagInfos'
  , 'softPFMuonsTagInfos'
  , 'softPFElectronsTagInfos'
    # C-Tagging tag infos
  , 'pfInclusiveSecondaryVertexFinderCvsLTagInfos'
  , 'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos'
    # TopTagInfos (unrelated to b tagging)
  , 'caTopTagInfos'
  ]
# extend for "internal use" in PAT/MINIAOD (renaming)
supportedBtagInfos.append( 'caTopTagInfosPAT' )

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                                  : []
    # legacy framework (no longer supported, work with RECO/AOD but not MiniAOD)
  , 'jetBProbabilityBJetTags'                               : ['impactParameterTagInfos']
  , 'jetProbabilityBJetTags'                                : ['impactParameterTagInfos']
  , 'positiveOnlyJetBProbabilityBJetTags'                   : ['impactParameterTagInfos']
  , 'positiveOnlyJetProbabilityBJetTags'                    : ['impactParameterTagInfos']
  , 'negativeOnlyJetBProbabilityBJetTags'                   : ['impactParameterTagInfos']
  , 'negativeOnlyJetProbabilityBJetTags'                    : ['impactParameterTagInfos']
  , 'trackCountingHighPurBJetTags'                          : ['impactParameterTagInfos']
  , 'trackCountingHighEffBJetTags'                          : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighPurBJetTags'                  : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighEffBJetTags'                  : ['impactParameterTagInfos']
  , 'simpleSecondaryVertexHighEffBJetTags'                  : ['secondaryVertexTagInfos']
  , 'simpleSecondaryVertexHighPurBJetTags'                  : ['secondaryVertexTagInfos']
  , 'negativeSimpleSecondaryVertexHighEffBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'negativeSimpleSecondaryVertexHighPurBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'combinedSecondaryVertexV2BJetTags'                     : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'positiveCombinedSecondaryVertexV2BJetTags'             : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'negativeCombinedSecondaryVertexV2BJetTags'             : ['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos']
  , 'simpleInclusiveSecondaryVertexHighEffBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'simpleInclusiveSecondaryVertexHighPurBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'negativeSimpleInclusiveSecondaryVertexHighEffBJetTags' : ['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos']
  , 'negativeSimpleInclusiveSecondaryVertexHighPurBJetTags' : ['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos']
  , 'doubleSecondaryVertexHighEffBJetTags'                  : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'combinedInclusiveSecondaryVertexV2BJetTags'            : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'positiveCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'negativeCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'combinedMVAV2BJetTags'                                 : ['impactParameterTagInfos', 'secondaryVertexTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'negativeCombinedMVAV2BJetTags'                         : ['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'positiveCombinedMVAV2BJetTags'                         : ['impactParameterTagInfos', 'secondaryVertexTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
    # new candidate-based framework (supported with RECO/AOD/MiniAOD)
  , 'pfJetBProbabilityBJetTags'                             : ['pfImpactParameterTagInfos']
  , 'pfJetProbabilityBJetTags'                              : ['pfImpactParameterTagInfos']
  , 'pfPositiveOnlyJetBProbabilityBJetTags'                 : ['pfImpactParameterTagInfos']
  , 'pfPositiveOnlyJetProbabilityBJetTags'                  : ['pfImpactParameterTagInfos']
  , 'pfNegativeOnlyJetBProbabilityBJetTags'                 : ['pfImpactParameterTagInfos']
  , 'pfNegativeOnlyJetProbabilityBJetTags'                  : ['pfImpactParameterTagInfos']
  , 'pfTrackCountingHighPurBJetTags'                        : ['pfImpactParameterTagInfos']
  , 'pfTrackCountingHighEffBJetTags'                        : ['pfImpactParameterTagInfos']
  , 'pfNegativeTrackCountingHighPurBJetTags'                : ['pfImpactParameterTagInfos']
  , 'pfNegativeTrackCountingHighEffBJetTags'                : ['pfImpactParameterTagInfos']
  , 'pfSimpleSecondaryVertexHighEffBJetTags'                : ['pfSecondaryVertexTagInfos']
  , 'pfSimpleSecondaryVertexHighPurBJetTags'                : ['pfSecondaryVertexTagInfos']
  , 'pfNegativeSimpleSecondaryVertexHighEffBJetTags'        : ['pfSecondaryVertexNegativeTagInfos']
  , 'pfNegativeSimpleSecondaryVertexHighPurBJetTags'        : ['pfSecondaryVertexNegativeTagInfos']
  , 'pfSimpleInclusiveSecondaryVertexHighEffBJetTags'       : ['pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfSimpleInclusiveSecondaryVertexHighPurBJetTags'       : ['pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags' : ['pfInclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags' : ['pfInclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'pfCombinedSecondaryVertexV2BJetTags'                   : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
  , 'pfPositiveCombinedSecondaryVertexV2BJetTags'           : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
  , 'pfNegativeCombinedSecondaryVertexV2BJetTags'           : ['pfImpactParameterTagInfos', 'pfSecondaryVertexNegativeTagInfos']
  , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'          : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags'  : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags'  : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'softPFMuonBJetTags'                                    : ['softPFMuonsTagInfos']
  , 'softPFMuonByPtBJetTags'                                : ['softPFMuonsTagInfos']
  , 'softPFMuonByIP3dBJetTags'                              : ['softPFMuonsTagInfos']
  , 'softPFMuonByIP2dBJetTags'                              : ['softPFMuonsTagInfos']
  , 'positiveSoftPFMuonBJetTags'                            : ['softPFMuonsTagInfos']
  , 'positiveSoftPFMuonByPtBJetTags'                        : ['softPFMuonsTagInfos']
  , 'positiveSoftPFMuonByIP3dBJetTags'                      : ['softPFMuonsTagInfos']
  , 'positiveSoftPFMuonByIP2dBJetTags'                      : ['softPFMuonsTagInfos']
  , 'negativeSoftPFMuonBJetTags'                            : ['softPFMuonsTagInfos']
  , 'negativeSoftPFMuonByPtBJetTags'                        : ['softPFMuonsTagInfos']
  , 'negativeSoftPFMuonByIP3dBJetTags'                      : ['softPFMuonsTagInfos']
  , 'negativeSoftPFMuonByIP2dBJetTags'                      : ['softPFMuonsTagInfos']
  , 'softPFElectronBJetTags'                                : ['softPFElectronsTagInfos']
  , 'softPFElectronByPtBJetTags'                            : ['softPFElectronsTagInfos']
  , 'softPFElectronByIP3dBJetTags'                          : ['softPFElectronsTagInfos']
  , 'softPFElectronByIP2dBJetTags'                          : ['softPFElectronsTagInfos']
  , 'positiveSoftPFElectronBJetTags'                        : ['softPFElectronsTagInfos']
  , 'positiveSoftPFElectronByPtBJetTags'                    : ['softPFElectronsTagInfos']
  , 'positiveSoftPFElectronByIP3dBJetTags'                  : ['softPFElectronsTagInfos']
  , 'positiveSoftPFElectronByIP2dBJetTags'                  : ['softPFElectronsTagInfos']
  , 'negativeSoftPFElectronBJetTags'                        : ['softPFElectronsTagInfos']
  , 'negativeSoftPFElectronByPtBJetTags'                    : ['softPFElectronsTagInfos']
  , 'negativeSoftPFElectronByIP3dBJetTags'                  : ['softPFElectronsTagInfos']
  , 'negativeSoftPFElectronByIP2dBJetTags'                  : ['softPFElectronsTagInfos']
  , 'pfCombinedMVAV2BJetTags'                               : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'pfNegativeCombinedMVAV2BJetTags'                       : ['pfImpactParameterTagInfos', 'pfSecondaryVertexNegativeTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'pfPositiveCombinedMVAV2BJetTags'                       : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'pfBoostedDoubleSecondaryVertexAK8BJetTags'             : ['pfImpactParameterAK8TagInfos', 'pfInclusiveSecondaryVertexFinderAK8TagInfos']
  , 'pfBoostedDoubleSecondaryVertexCA15BJetTags'            : ['pfImpactParameterCA15TagInfos', 'pfInclusiveSecondaryVertexFinderCA15TagInfos']
    # C-Tagging
  , 'pfCombinedCvsLJetTags'                                 : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
  , 'pfNegativeCombinedCvsLJetTags'                         : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
  , 'pfPositiveCombinedCvsLJetTags'                         : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
  , 'pfCombinedCvsBJetTags'                                 : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
  , 'pfNegativeCombinedCvsBJetTags'                         : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]  
  , 'pfPositiveCombinedCvsBJetTags'                         : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
    # ChargeTagging
  , 'pfChargeBJetTags'                                      : ["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]
  }
