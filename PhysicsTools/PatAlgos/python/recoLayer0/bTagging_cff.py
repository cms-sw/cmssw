#add negative secondary vertex tagger

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex2TrkES_cfi import *
#from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
#from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeBJetTags_cfi import *


## list of all available btagInfos
supportedBtagInfos = [
    'None',
    'impactParameterTagInfos',
    'secondaryVertexTagInfos',
    'softMuonTagInfos',
    'secondaryVertexNegativeTagInfos',
    ]

## dictionary with all available btag discriminators and the btagInfos that they require
supportedBtagDiscr = {
    'None'                                         : [],
    'jetBProbabilityBJetTags'                      : ['impactParameterTagInfos'],
    'jetProbabilityBJetTags'                       : ['impactParameterTagInfos'],
    'trackCountingHighPurBJetTags'                 : ['impactParameterTagInfos'],
    'trackCountingHighEffBJetTags'                 : ['impactParameterTagInfos'],
    'simpleSecondaryVertexHighEffBJetTags'         : ['secondaryVertexTagInfos'],
    'simpleSecondaryVertexHighPurBJetTags'         : ['secondaryVertexTagInfos'],
    'combinedSecondaryVertexBJetTags'              : ['impactParameterTagInfos', 'secondaryVertexTagInfos'],
    'combinedSecondaryVertexMVABJetTags'           : ['impactParameterTagInfos', 'secondaryVertexTagInfos'],
    'softMuonBJetTags'                             : ['softMuonTagInfos'],
    'softMuonByPtBJetTags'                         : ['softMuonTagInfos'],
    'softMuonByIP3dBJetTags'                       : ['softMuonTagInfos'],
    'simpleSecondaryVertexNegativeHighEffBJetTags' : ['secondaryVertexNegativeTagInfos'],
    'simpleSecondaryVertexNegativeHighPurBJetTags' : ['secondaryVertexNegativeTagInfos'],
    'negativeTrackCountingHighEffJetTags'          : ['impactParameterTagInfos'],
    'negativeTrackCountingHighPurJetTags'          : ['impactParameterTagInfos'],
    }
