## list of all available btagInfos
supportedBtagInfos = [
    'None'
  , 'pfImpactParameterTagInfos'
  , 'pfSecondaryVertexTagInfos'
  , 'pfInclusiveSecondaryVertexFinderTagInfos'
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
  , 'caTopTagInfos'
  ]

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                                  : []
  , 'pfJetBProbabilityBJetTags'                             : ['pfImpactParameterTagInfos']
  , 'pfJetProbabilityBJetTags'                              : ['pfImpactParameterTagInfos']
  , 'jetBProbabilityBJetTags'                               : ['impactParameterTagInfos']
  , 'jetProbabilityBJetTags'                                : ['impactParameterTagInfos']
  , 'positiveOnlyJetBProbabilityBJetTags'                   : ['impactParameterTagInfos']
  , 'positiveOnlyJetProbabilityBJetTags'                    : ['impactParameterTagInfos']
  , 'negativeOnlyJetBProbabilityBJetTags'                   : ['impactParameterTagInfos']
  , 'negativeOnlyJetProbabilityBJetTags'                    : ['impactParameterTagInfos']
  , 'pfTrackCountingHighPurBJetTags'                        : ['pfImpactParameterTagInfos']
  , 'pfTrackCountingHighEffBJetTags'                        : ['pfImpactParameterTagInfos']
  , 'trackCountingHighPurBJetTags'                          : ['impactParameterTagInfos']
  , 'trackCountingHighEffBJetTags'                          : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighPurBJetTags'                  : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighEffBJetTags'                  : ['impactParameterTagInfos']
  , 'pfSimpleSecondaryVertexHighEffBJetTags'                : ['pfSecondaryVertexTagInfos']
  , 'pfSimpleSecondaryVertexHighPurBJetTags'                : ['pfSecondaryVertexTagInfos']
  , 'simpleSecondaryVertexHighEffBJetTags'                  : ['secondaryVertexTagInfos']
  , 'simpleSecondaryVertexHighPurBJetTags'                  : ['secondaryVertexTagInfos']
  , 'negativeSimpleSecondaryVertexHighEffBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'negativeSimpleSecondaryVertexHighPurBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'pfCombinedSecondaryVertexBJetTags'                     : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
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
  , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'          : ['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']
  , 'combinedInclusiveSecondaryVertexV2BJetTags'            : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'positiveCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'negativeCombinedInclusiveSecondaryVertexV2BJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'combinedSecondaryVertexMVABJetTags'                    : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
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
  }
