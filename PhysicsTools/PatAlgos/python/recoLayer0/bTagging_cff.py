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
  , 'pfGhostTrackVertexTagInfos'
  , 'pfBoostedDoubleSVAK8TagInfos'
  , 'pfBoostedDoubleSVCA15TagInfos'
  , 'softPFMuonsTagInfos'
  , 'softPFElectronsTagInfos'
    # C-Tagging tag infos
  , 'pfInclusiveSecondaryVertexFinderCvsLTagInfos'
  , 'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos'
    # DeepFlavour	
  , 'pfDeepCSVTagInfos'
  , 'pfDeepCSVNegativeTagInfos'
  , 'pfDeepCSVPositiveTagInfos'
    # DeepCMVA	
  , 'pfDeepCMVATagInfos'
  , 'pfDeepCMVANegativeTagInfos'
  , 'pfDeepCMVAPositiveTagInfos'
    # TopTagInfos (unrelated to b tagging)
  , 'caTopTagInfos'
  ]
# extend for "internal use" in PAT/MINIAOD (renaming)
supportedBtagInfos.append( 'caTopTagInfosPAT' )

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                                  : []
    # legacy framework (no longer supported, work with RECO/AOD but not MiniAOD)
  , 'jetBProbabilityBJetTags'                               : [['impactParameterTagInfos']]
  , 'jetProbabilityBJetTags'                                : [['impactParameterTagInfos']]
  , 'positiveOnlyJetBProbabilityBJetTags'                   : [['impactParameterTagInfos']]
  , 'positiveOnlyJetProbabilityBJetTags'                    : [['impactParameterTagInfos']]
  , 'negativeOnlyJetBProbabilityBJetTags'                   : [['impactParameterTagInfos']]
  , 'negativeOnlyJetProbabilityBJetTags'                    : [['impactParameterTagInfos']]
  , 'trackCountingHighPurBJetTags'                          : [['impactParameterTagInfos']]
  , 'trackCountingHighEffBJetTags'                          : [['impactParameterTagInfos']]
  , 'negativeTrackCountingHighPurBJetTags'                  : [['impactParameterTagInfos']]
  , 'negativeTrackCountingHighEffBJetTags'                  : [['impactParameterTagInfos']]
  , 'simpleSecondaryVertexHighEffBJetTags'                  : [['secondaryVertexTagInfos'], ['impactParameterTagInfos']]
  , 'simpleSecondaryVertexHighPurBJetTags'                  : [['secondaryVertexTagInfos'], ['impactParameterTagInfos']]
  , 'negativeSimpleSecondaryVertexHighEffBJetTags'          : [['secondaryVertexNegativeTagInfos'], ['impactParameterTagInfos']]
  , 'negativeSimpleSecondaryVertexHighPurBJetTags'          : [['secondaryVertexNegativeTagInfos'], ['impactParameterTagInfos']]
  , 'combinedSecondaryVertexV2BJetTags'                     : [['impactParameterTagInfos', 'secondaryVertexTagInfos']]
  , 'positiveCombinedSecondaryVertexV2BJetTags'             : [['impactParameterTagInfos', 'secondaryVertexTagInfos']]
  , 'negativeCombinedSecondaryVertexV2BJetTags'             : [['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos']]
  , 'simpleInclusiveSecondaryVertexHighEffBJetTags'         : [['inclusiveSecondaryVertexFinderFilteredTagInfos'], ['impactParameterTagInfos']]
  , 'simpleInclusiveSecondaryVertexHighPurBJetTags'         : [['inclusiveSecondaryVertexFinderFilteredTagInfos'], ['impactParameterTagInfos']]
  , 'negativeSimpleInclusiveSecondaryVertexHighEffBJetTags' : [['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos'], ['impactParameterTagInfos']]
  , 'negativeSimpleInclusiveSecondaryVertexHighPurBJetTags' : [['inclusiveSecondaryVertexFinderFilteredNegativeTagInfos'], ['impactParameterTagInfos']]
  , 'doubleSecondaryVertexHighEffBJetTags'                  : [['inclusiveSecondaryVertexFinderFilteredTagInfos'], ['impactParameterTagInfos']]
  , 'combinedInclusiveSecondaryVertexV2BJetTags'            : [['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']]
  , 'positiveCombinedInclusiveSecondaryVertexV2BJetTags'    : [['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']]
  , 'negativeCombinedInclusiveSecondaryVertexV2BJetTags'    : [['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'combinedMVAV2BJetTags'                                 : [['impactParameterTagInfos', 'secondaryVertexTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
  , 'negativeCombinedMVAV2BJetTags'                         : [['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
  , 'positiveCombinedMVAV2BJetTags'                         : [['impactParameterTagInfos', 'secondaryVertexTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
    # new candidate-based framework (supported with RECO/AOD/MiniAOD)
  , 'pfJetBProbabilityBJetTags'                             : [['pfImpactParameterTagInfos']]
  , 'pfJetProbabilityBJetTags'                              : [['pfImpactParameterTagInfos']]
  , 'pfPositiveOnlyJetBProbabilityBJetTags'                 : [['pfImpactParameterTagInfos']]
  , 'pfPositiveOnlyJetProbabilityBJetTags'                  : [['pfImpactParameterTagInfos']]
  , 'pfNegativeOnlyJetBProbabilityBJetTags'                 : [['pfImpactParameterTagInfos']]
  , 'pfNegativeOnlyJetProbabilityBJetTags'                  : [['pfImpactParameterTagInfos']]
  , 'pfTrackCountingHighPurBJetTags'                        : [['pfImpactParameterTagInfos']]
  , 'pfTrackCountingHighEffBJetTags'                        : [['pfImpactParameterTagInfos']]
  , 'pfNegativeTrackCountingHighPurBJetTags'                : [['pfImpactParameterTagInfos']]
  , 'pfNegativeTrackCountingHighEffBJetTags'                : [['pfImpactParameterTagInfos']]
  , 'pfSimpleSecondaryVertexHighEffBJetTags'                : [['pfSecondaryVertexTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfSimpleSecondaryVertexHighPurBJetTags'                : [['pfSecondaryVertexTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfNegativeSimpleSecondaryVertexHighEffBJetTags'        : [['pfSecondaryVertexNegativeTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfNegativeSimpleSecondaryVertexHighPurBJetTags'        : [['pfSecondaryVertexNegativeTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfSimpleInclusiveSecondaryVertexHighEffBJetTags'       : [['pfInclusiveSecondaryVertexFinderTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfSimpleInclusiveSecondaryVertexHighPurBJetTags'       : [['pfInclusiveSecondaryVertexFinderTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags' : [['pfInclusiveSecondaryVertexFinderNegativeTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags' : [['pfInclusiveSecondaryVertexFinderNegativeTagInfos'], ['pfImpactParameterTagInfos']]
  , 'pfCombinedSecondaryVertexV2BJetTags'                   : [['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']]
  , 'pfPositiveCombinedSecondaryVertexV2BJetTags'           : [['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']]
  , 'pfNegativeCombinedSecondaryVertexV2BJetTags'           : [['pfImpactParameterTagInfos', 'pfSecondaryVertexNegativeTagInfos']]
  , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'          : [['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags'  : [['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags'  : [['pfImpactParameterTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfGhostTrackBJetTags'                                  : [['pfImpactParameterTagInfos', 'pfGhostTrackVertexTagInfos']]
  , 'softPFMuonBJetTags'                                    : [['softPFMuonsTagInfos']]
  , 'softPFMuonByPtBJetTags'                                : [['softPFMuonsTagInfos']]
  , 'softPFMuonByIP3dBJetTags'                              : [['softPFMuonsTagInfos']]
  , 'softPFMuonByIP2dBJetTags'                              : [['softPFMuonsTagInfos']]
  , 'positiveSoftPFMuonBJetTags'                            : [['softPFMuonsTagInfos']]
  , 'positiveSoftPFMuonByPtBJetTags'                        : [['softPFMuonsTagInfos']]
  , 'positiveSoftPFMuonByIP3dBJetTags'                      : [['softPFMuonsTagInfos']]
  , 'positiveSoftPFMuonByIP2dBJetTags'                      : [['softPFMuonsTagInfos']]
  , 'negativeSoftPFMuonBJetTags'                            : [['softPFMuonsTagInfos']]
  , 'negativeSoftPFMuonByPtBJetTags'                        : [['softPFMuonsTagInfos']]
  , 'negativeSoftPFMuonByIP3dBJetTags'                      : [['softPFMuonsTagInfos']]
  , 'negativeSoftPFMuonByIP2dBJetTags'                      : [['softPFMuonsTagInfos']]
  , 'softPFElectronBJetTags'                                : [['softPFElectronsTagInfos']]
  , 'softPFElectronByPtBJetTags'                            : [['softPFElectronsTagInfos']]
  , 'softPFElectronByIP3dBJetTags'                          : [['softPFElectronsTagInfos']]
  , 'softPFElectronByIP2dBJetTags'                          : [['softPFElectronsTagInfos']]
  , 'positiveSoftPFElectronBJetTags'                        : [['softPFElectronsTagInfos']]
  , 'positiveSoftPFElectronByPtBJetTags'                    : [['softPFElectronsTagInfos']]
  , 'positiveSoftPFElectronByIP3dBJetTags'                  : [['softPFElectronsTagInfos']]
  , 'positiveSoftPFElectronByIP2dBJetTags'                  : [['softPFElectronsTagInfos']]
  , 'negativeSoftPFElectronBJetTags'                        : [['softPFElectronsTagInfos']]
  , 'negativeSoftPFElectronByPtBJetTags'                    : [['softPFElectronsTagInfos']]
  , 'negativeSoftPFElectronByIP3dBJetTags'                  : [['softPFElectronsTagInfos']]
  , 'negativeSoftPFElectronByIP2dBJetTags'                  : [['softPFElectronsTagInfos']]
  , 'pfCombinedMVAV2BJetTags'                               : [['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
  , 'pfNegativeCombinedMVAV2BJetTags'                       : [['pfImpactParameterTagInfos', 'pfSecondaryVertexNegativeTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
  , 'pfPositiveCombinedMVAV2BJetTags'                       : [['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']]
  , 'pfBoostedDoubleSecondaryVertexAK8BJetTags'             : [['pfBoostedDoubleSVAK8TagInfos'], ['pfImpactParameterAK8TagInfos', 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfBoostedDoubleSecondaryVertexCA15BJetTags'            : [['pfBoostedDoubleSVCA15TagInfos'], ['pfImpactParameterCA15TagInfos', 'pfInclusiveSecondaryVertexFinderCA15TagInfos']]
    # C-Tagging
  , 'pfCombinedCvsLJetTags'                                 : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  , 'pfNegativeCombinedCvsLJetTags'                         : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  , 'pfPositiveCombinedCvsLJetTags'                         : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  , 'pfCombinedCvsBJetTags'                                 : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  , 'pfNegativeCombinedCvsBJetTags'                         : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  , 'pfPositiveCombinedCvsBJetTags'                         : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderCvsLTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
    # DeepFlavour
  , 'pfDeepCSVJetTags:probudsg'                           : [['pfDeepCSVTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCSVJetTags:probb'                              : [['pfDeepCSVTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCSVJetTags:probc'                              : [['pfDeepCSVTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCSVJetTags:probbb'                             : [['pfDeepCSVTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCSVJetTags:probcc'                             : [['pfDeepCSVTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfNegativeDeepCSVJetTags:probudsg'                   : [['pfDeepCSVNegativeTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepCSVJetTags:probb'                      : [['pfDeepCSVNegativeTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepCSVJetTags:probc'                      : [['pfDeepCSVNegativeTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepCSVJetTags:probbb'                     : [['pfDeepCSVNegativeTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepCSVJetTags:probcc'                     : [['pfDeepCSVNegativeTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfPositiveDeepCSVJetTags:probudsg'                   : [['pfDeepCSVPositiveTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfPositiveDeepCSVJetTags:probb'                      : [['pfDeepCSVPositiveTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfPositiveDeepCSVJetTags:probc'                      : [['pfDeepCSVPositiveTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfPositiveDeepCSVJetTags:probbb'                     : [['pfDeepCSVPositiveTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfPositiveDeepCSVJetTags:probcc'                     : [['pfDeepCSVPositiveTagInfos'], ["pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
    # DeepCMVA
  , 'pfDeepCMVAJetTags:probudsg'                           : [["pfDeepCMVATagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfDeepCMVAJetTags:probb'                              : [["pfDeepCMVATagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfDeepCMVAJetTags:probc'                              : [["pfDeepCMVATagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfDeepCMVAJetTags:probbb'                             : [["pfDeepCMVATagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfDeepCMVAJetTags:probcc'                             : [["pfDeepCMVATagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfNegativeDeepCMVAJetTags:probudsg'                   : [["pfDeepCMVANegativeTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderNegativeTagInfos"]]
  , 'pfNegativeDeepCMVAJetTags:probb'                      : [["pfDeepCMVANegativeTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderNegativeTagInfos"]]
  , 'pfNegativeDeepCMVAJetTags:probc'                      : [["pfDeepCMVANegativeTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderNegativeTagInfos"]]
  , 'pfNegativeDeepCMVAJetTags:probbb'                     : [["pfDeepCMVANegativeTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderNegativeTagInfos"]]
  , 'pfNegativeDeepCMVAJetTags:probcc'                     : [["pfDeepCMVANegativeTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderNegativeTagInfos"]]
  , 'pfPositiveDeepCMVAJetTags:probudsg'                   : [["pfDeepCMVAPositiveTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfPositiveDeepCMVAJetTags:probb'                      : [["pfDeepCMVAPositiveTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfPositiveDeepCMVAJetTags:probc'                      : [["pfDeepCMVAPositiveTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfPositiveDeepCMVAJetTags:probbb'                     : [["pfDeepCMVAPositiveTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]
  , 'pfPositiveDeepCMVAJetTags:probcc'                     : [["pfDeepCMVAPositiveTagInfos"], ["pfImpactParameterTagInfos","softPFMuonsTagInfos","softPFElectronsTagInfos","pfInclusiveSecondaryVertexFinderTagInfos"]]    
    # ChargeTagging
  , 'pfChargeBJetTags'                                      : [["pfImpactParameterTagInfos", "pfInclusiveSecondaryVertexFinderTagInfos", "softPFMuonsTagInfos", "softPFElectronsTagInfos"]]
  }
