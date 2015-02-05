## list of all available btagInfos
supportedBtagInfos = [
    'None'
    # legacy framework (not supported with MiniAOD)
  , 'impactParameterTagInfos'
  , 'secondaryVertexTagInfos'
  , 'secondaryVertexNegativeTagInfos'
  , 'softMuonTagInfos'
  , 'softPFMuonsTagInfos'
  , 'softPFElectronsTagInfos'
  , 'inclusiveSecondaryVertexFinderTagInfos'
  , 'inclusiveSecondaryVertexFinderNegativeTagInfos'
  , 'inclusiveSecondaryVertexFinderFilteredTagInfos'
  , 'inclusiveSecondaryVertexFinderFilteredNegativeTagInfos'
     # new candidate-based framework (supported with RECO/AOD/MiniAOD)
  , 'pfImpactParameterTagInfos'
  , 'pfSecondaryVertexTagInfos'
  , 'pfSecondaryVertexNegativeTagInfos'
  , 'pfInclusiveSecondaryVertexFinderTagInfos'
  , 'pfInclusiveSecondaryVertexFinderNegativeTagInfos'
  #, 'pfInclusiveSecondaryVertexFinderCtagLTagInfos'
  #, 'pfInclusiveSecondaryVertexFinderCtagLNegativeTagInfos'
  , 'caTopTagInfos'
  ]
# extend for "internal use" in PAT/MINIAOD (renaming)
supportedBtagInfos.append( 'caTopTagInfosPAT' )

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                                  : []
    # legacy framework (not supported with MiniAOD)
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
  , 'combinedSecondaryVertexBJetTags'                       : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'positiveCombinedSecondaryVertexBJetTags'               : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'negativeCombinedSecondaryVertexBJetTags'               : ['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos']
  , 'simpleInclusiveSecondaryVertexHighEffBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'simpleInclusiveSecondaryVertexHighPurBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'negativeSimpleInclusiveSecondaryVertexHighEffBJetTags' : ['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos']
  , 'negativeSimpleInclusiveSecondaryVertexHighPurBJetTags' : ['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos']
  , 'doubleSecondaryVertexHighEffBJetTags'                  : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'combinedInclusiveSecondaryVertexBJetTags'              : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'positiveCombinedInclusiveSecondaryVertexBJetTags'      : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'negativeCombinedInclusiveSecondaryVertexBJetTags'      : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'combinedInclusiveSecondaryVertexV2BJetTags'            : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'positiveCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'negativeCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']
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
  , 'combinedMVABJetTags'                                   : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'positiveCombinedMVABJetTags'                           : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'negativeCombinedMVABJetTags'                           : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
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
  , 'pfCombinedSecondaryVertexBJetTags'                     : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
  , 'pfPositiveCombinedSecondaryVertexBJetTags'             : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
  , 'pfNegativeCombinedSecondaryVertexBJetTags'             : ['pfImpactParameterTagInfos', 'pfSecondaryVertexNegativeTagInfos']
  , 'pfCombinedInclusiveSecondaryVertexBJetTags'            : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfPositiveCombinedInclusiveSecondaryVertexBJetTags'    : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfNegativeCombinedInclusiveSecondaryVertexBJetTags'    : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'          : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags'  : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags'  : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'pfCombinedMVABJetTags'                                 : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'pfPositiveCombinedMVABJetTags'                         : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'pfNegativeCombinedMVABJetTags'                         : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'pfCombinedSecondaryVertexSoftLeptonBJetTags'           : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'pfNegativeCombinedSecondaryVertexSoftLeptonBJetTags'   : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'pfCombinedSecondaryVertexSoftLeptonCtagLJetTags'       : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderCtagLTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'pfNegativeCombinedSecondaryVertexSoftLeptonCtagLJetTags' : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderCtagLNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  }
