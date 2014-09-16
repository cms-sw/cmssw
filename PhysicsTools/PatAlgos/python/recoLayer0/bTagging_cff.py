#add negative secondary vertex tagger

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex2TrkES_cfi import *
#from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
#from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeBJetTags_cfi import *


## list of all available btagInfos
supportedBtagInfos = [
    'None'
  , 'pfImpactParameterTagInfos'
  , 'pfSecondaryVertexTagInfos'
  , 'impactParameterTagInfos'
  , 'secondaryVertexTagInfos'
  , 'secondaryVertexNegativeTagInfos'
  , 'softMuonTagInfos'
  , 'softPFMuonsTagInfos'
  , 'softPFElectronsTagInfos'
  , 'inclusiveSecondaryVertexFinderTagInfos'
  , 'inclusiveSecondaryVertexFinderNegativeTagInfos'
  , 'inclusiveSecondaryVertexFinderFilteredTagInfos'
  , 'caTopTagInfos'
  ]

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                                  : []
  , 'jetBProbabilityBJetTags'                               : ['impactParameterTagInfos']
  , 'jetProbabilityBJetTags'                                : ['impactParameterTagInfos']
  , 'trackCountingHighPurBJetTags'                          : ['impactParameterTagInfos']
  , 'trackCountingHighEffBJetTags'                          : ['impactParameterTagInfos']
  , 'negativeOnlyJetBProbabilityJetTags'                    : ['impactParameterTagInfos']
  , 'negativeOnlyJetProbabilityJetTags'                     : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighEffJetTags'                   : ['impactParameterTagInfos']
  , 'negativeTrackCountingHighPurJetTags'                   : ['impactParameterTagInfos']
  , 'positiveOnlyJetBProbabilityJetTags'                    : ['impactParameterTagInfos']
  , 'positiveOnlyJetProbabilityJetTags'                     : ['impactParameterTagInfos']
  , 'simpleSecondaryVertexHighEffBJetTags'                  : ['secondaryVertexTagInfos']
  , 'simpleSecondaryVertexHighPurBJetTags'                  : ['secondaryVertexTagInfos']
  , 'simpleSecondaryVertexNegativeHighEffBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'simpleSecondaryVertexNegativeHighPurBJetTags'          : ['secondaryVertexNegativeTagInfos']
  , 'pfCombinedSecondaryVertexBJetTags'                     : ['pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos']
  , 'combinedSecondaryVertexBJetTags'                       : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'combinedSecondaryVertexPositiveBJetTags'               : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'combinedInclusiveSecondaryVertexV2BJetTags'            : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'combinedInclusiveSecondaryVertexV2PositiveBJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'combinedInclusiveSecondaryVertexV2NegativeBJetTags'    : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderNegativeTagInfos']
  , 'combinedSecondaryVertexMVABJetTags'                    : ['impactParameterTagInfos', 'secondaryVertexTagInfos']
  , 'combinedSecondaryVertexNegativeBJetTags'               : ['impactParameterTagInfos', 'secondaryVertexNegativeTagInfos']
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
  , 'simpleInclusiveSecondaryVertexHighEffBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'simpleInclusiveSecondaryVertexHighPurBJetTags'         : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'doubleSecondaryVertexHighEffBJetTags'                  : ['inclusiveSecondaryVertexFinderFilteredTagInfos']
  , 'combinedInclusiveSecondaryVertexBJetTags'              : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  , 'combinedInclusiveSecondaryVertexPositiveBJetTags'      : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos']
  #, 'combinedMVABJetTags'                                   : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'positiveCombinedMVABJetTags'                           : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  , 'negativeCombinedMVABJetTags'                           : ['impactParameterTagInfos', 'inclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'combinedSecondaryVertexSoftPFLeptonV1BJetTags'         : ['impactParameterTagInfos', 'secondaryVertexTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'positiveCombinedSecondaryVertexSoftPFLeptonV1BJetTags' : ['impactParameterTagInfos', 'secondaryVertexTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  #, 'negativeCombinedSecondaryVertexSoftPFLeptonV1BJetTags' : ['impactParameterTagInfos', 'secondaryVertexTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos']
  }
