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
    # DeepCSV
  , 'pfDeepCSVTagInfos'
  , 'pfDeepCSVNegativeTagInfos'
  , 'pfDeepCSVPositiveTagInfos'
    # DeepCMVA	
  , 'pfDeepCMVATagInfos'
  , 'pfDeepCMVANegativeTagInfos'
  , 'pfDeepCMVAPositiveTagInfos'
    # TopTagInfos (unrelated to b tagging)
  , 'caTopTagInfos'
    # DeepFlavour tag infos
  , 'pfDeepFlavourTagInfos'
  , 'pfNegativeDeepFlavourTagInfos'
    # DeepDoubleB/C tag infos
  , 'pfDeepDoubleXTagInfos'
    # DeepBoostedJet tag infos
  , 'pfDeepBoostedJetTagInfos'
    # ParticleNet (AK8) tag infos
  , 'pfParticleNetTagInfos'
    # ParticleNet (AK4) tag infos
  , 'pfParticleNetAK4TagInfos'
  , 'pfNegativeParticleNetAK4TagInfos'
    # Pixel Cluster tag infos
  , 'pixelClusterTagInfos'
    # HiggsInteractionNet tag infos
  , 'pfHiggsInteractionNetTagInfos'
  , 'pfParticleNetFromMiniAODAK4PuppiCentralTagInfos'
  , 'pfParticleNetFromMiniAODAK4PuppiForwardTagInfos'
  , 'pfParticleNetFromMiniAODAK4CHSCentralTagInfos'
  , 'pfParticleNetFromMiniAODAK4CHSForwardTagInfos'
  , 'pfParticleNetFromMiniAODAK8TagInfos'
 
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
  , 'pfDeepFlavourJetTags:probb'                            : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepFlavourJetTags:probbb'                           : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepFlavourJetTags:problepb'                         : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepFlavourJetTags:probc'                            : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepFlavourJetTags:probuds'                          : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepFlavourJetTags:probg'                            : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepVertexJetTags:probb'                             : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCombinedJetTags:probb'                           : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCombinedJetTags:probc'                           : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCombinedJetTags:probuds'                         : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfDeepCombinedJetTags:probg'                           : [["pfDeepFlavourTagInfos"], ['pfDeepCSVTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:probb'                            : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:probbb'                           : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:problepb'                         : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:probc'                            : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:probuds'                          : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfNegativeDeepFlavourJetTags:probg'                            : [["pfNegativeDeepFlavourTagInfos"], ['pfDeepCSVNegativeTagInfos', "pfImpactParameterTagInfos", 'pfInclusiveSecondaryVertexFinderNegativeTagInfos']]
  , 'pfDeepDoubleBvLJetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfDeepDoubleBvLJetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfDeepDoubleCvLJetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfDeepDoubleCvLJetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfDeepDoubleCvBJetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfDeepDoubleCvBJetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleBvLJetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleBvLJetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvLJetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvLJetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvBJetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvBJetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleBvLV2JetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleBvLV2JetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvLV2JetTags:probQCD'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvLV2JetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvBV2JetTags:probHbb'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
  , 'pfMassIndependentDeepDoubleCvBV2JetTags:probHcc'                     : [["pfDeepDoubleXTagInfos"], ['pfBoostedDoubleSVAK8TagInfos', "pfImpactParameterAK8TagInfos", 'pfInclusiveSecondaryVertexFinderAK8TagInfos']]
}

# meta-taggers are simple arithmetic on top of other taggers, they are stored here
# such that in case you want them re-run also the parent tagger is re-run as well

supportedMetaDiscr = {
   'pfDeepCSVDiscriminatorsJetTags:BvsAll' : ['pfDeepCSVJetTags:probudsg', 'pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probc', 'pfDeepCSVJetTags:probbb'],
   'pfDeepCSVDiscriminatorsJetTags:CvsB' : ['pfDeepCSVJetTags:probudsg', 'pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probc', 'pfDeepCSVJetTags:probbb'],
   'pfDeepCSVDiscriminatorsJetTags:CvsL' : ['pfDeepCSVJetTags:probudsg', 'pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probc', 'pfDeepCSVJetTags:probbb'],
   'pfDeepCMVADiscriminatorsJetTags:BvsAll' : ['pfDeepCMVAJetTags:probudsg', 'pfDeepCMVAJetTags:probb', 'pfDeepCMVAJetTags:probc', 'pfDeepCMVAJetTags:probbb'],
   'pfDeepCMVADiscriminatorsJetTags:CvsB' : ['pfDeepCMVAJetTags:probudsg', 'pfDeepCMVAJetTags:probb', 'pfDeepCMVAJetTags:probc', 'pfDeepCMVAJetTags:probbb'],
   'pfDeepCMVADiscriminatorsJetTags:CvsL' : ['pfDeepCMVAJetTags:probudsg', 'pfDeepCMVAJetTags:probb', 'pfDeepCMVAJetTags:probc', 'pfDeepCMVAJetTags:probbb'],
}

# -----------------------------------
# setup DeepBoostedJet
from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsProbs, _pfDeepBoostedJetTagsMetaDiscrs, \
    _pfMassDecorrelatedDeepBoostedJetTagsProbs, _pfMassDecorrelatedDeepBoostedJetTagsMetaDiscrs
# update supportedBtagDiscr
for disc in _pfDeepBoostedJetTagsProbs + _pfMassDecorrelatedDeepBoostedJetTagsProbs:
    supportedBtagDiscr[disc] = [["pfDeepBoostedJetTagInfos"]]
# update supportedMetaDiscr
for disc in _pfDeepBoostedJetTagsMetaDiscrs:
    supportedMetaDiscr[disc] = _pfDeepBoostedJetTagsProbs
for disc in _pfMassDecorrelatedDeepBoostedJetTagsMetaDiscrs:
    supportedMetaDiscr[disc] = _pfMassDecorrelatedDeepBoostedJetTagsProbs
# -----------------------------------

# -----------------------------------
# setup ParticleNet AK8
from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsProbs, _pfParticleNetJetTagsMetaDiscrs, \
    _pfMassDecorrelatedParticleNetJetTagsProbs, _pfMassDecorrelatedParticleNetJetTagsMetaDiscrs, \
    _pfParticleNetMassRegressionOutputs
# update supportedBtagDiscr
for disc in _pfParticleNetJetTagsProbs + _pfMassDecorrelatedParticleNetJetTagsProbs + _pfParticleNetMassRegressionOutputs:
    supportedBtagDiscr[disc] = [["pfParticleNetTagInfos"]]
# update supportedMetaDiscr
for disc in _pfParticleNetJetTagsMetaDiscrs:
    supportedMetaDiscr[disc] = _pfParticleNetJetTagsProbs
for disc in _pfMassDecorrelatedParticleNetJetTagsMetaDiscrs:
    supportedMetaDiscr[disc] = _pfMassDecorrelatedParticleNetJetTagsProbs
# -----------------------------------

# -----------------------------------
# setup ParticleNet AK4
from RecoBTag.ONNXRuntime.pfParticleNetAK4_cff import _pfParticleNetAK4JetTagsProbs, _pfParticleNetAK4JetTagsMetaDiscrs
# update supportedBtagDiscr
for disc in _pfParticleNetAK4JetTagsProbs + _pfParticleNetAK4JetTagsMetaDiscrs:
    supportedBtagDiscr[disc] = [["pfParticleNetAK4TagInfos"]]
# update supportedMetaDiscr
for disc in _pfParticleNetAK4JetTagsMetaDiscrs:
    supportedMetaDiscr[disc] = _pfParticleNetAK4JetTagsProbs
# -----------------------------------
# setup Negative ParticleNet AK4
from RecoBTag.ONNXRuntime.pfParticleNetAK4_cff import _pfNegativeParticleNetAK4JetTagsProbs
for disc in _pfNegativeParticleNetAK4JetTagsProbs:
    supportedBtagDiscr[disc] = [["pfNegativeParticleNetAK4TagInfos"]]
# -----------------------------------
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs,_pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr
for disc in _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs:
    supportedBtagDiscr[disc] =  [["pfParticleNetFromMiniAODAK4PuppiCentralTagInfos"]]
for disc in _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr:
    supportedMetaDiscr[disc] = _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsProbs,_pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr
for disc in _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsProbs:
    supportedBtagDiscr[disc] =  [["pfParticleNetFromMiniAODAK4PuppiForwardTagInfos"]]
for disc in _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr:
    supportedMetaDiscr[disc] = _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsProbs
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSCentralJetTagsProbs,_pfParticleNetFromMiniAODAK4CHSCentralJetTagsMetaDiscr
for disc in _pfParticleNetFromMiniAODAK4CHSCentralJetTagsProbs:
    supportedBtagDiscr[disc] =  [["pfParticleNetFromMiniAODAK4CHSCentralTagInfos"]]
for disc in _pfParticleNetFromMiniAODAK4CHSCentralJetTagsMetaDiscr:
    supportedMetaDiscr[disc] = _pfParticleNetFromMiniAODAK4CHSCentralJetTagsProbs
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSForwardJetTagsProbs,_pfParticleNetFromMiniAODAK4CHSForwardJetTagsMetaDiscr
for disc in _pfParticleNetFromMiniAODAK4CHSForwardJetTagsProbs:
    supportedBtagDiscr[disc] =  [["pfParticleNetFromMiniAODAK4CHSForwardTagInfos"]]
for disc in _pfParticleNetFromMiniAODAK4CHSForwardJetTagsMetaDiscr:
    supportedMetaDiscr[disc] = _pfParticleNetFromMiniAODAK4CHSForwardJetTagsProbs
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsProbs,_pfParticleNetFromMiniAODAK8JetTagsMetaDiscr
for disc in _pfParticleNetFromMiniAODAK8JetTagsProbs:
    supportedBtagDiscr[disc] =  [["pfParticleNetFromMiniAODAK8TagInfos"]]
for disc in _pfParticleNetFromMiniAODAK8JetTagsMetaDiscr:
    supportedMetaDiscr[disc] = _pfParticleNetFromMiniAODAK8JetTagsProbs


# -----------------------------------
# setup HiggsInteractionNet
from RecoBTag.ONNXRuntime.pfHiggsInteractionNet_cff import _pfHiggsInteractionNetTagsProbs
# update supportedBtagDiscr 
for disc in _pfHiggsInteractionNetTagsProbs:
    supportedBtagDiscr[disc] = [["pfHiggsInteractionNetTagInfos"]]
# -----------------------------------
