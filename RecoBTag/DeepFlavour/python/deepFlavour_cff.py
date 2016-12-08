import FWCore.ParameterSet.Config as cms
from RecoBTag.DeepFlavour.DeepNNTagInfoProducer_cfi import deepNNTagInfos
from RecoBTag.DeepFlavour.DeepCMVATagInfoProducer_cfi import deepCMVATagInfos
from RecoBTag.DeepFlavour.DeepFlavourJetTagsProducer_cfi import deepFlavourJetTags, deepFlavourCMVAJetTags


##
## Negative and positive taggers for light SF estimation
##

deepNNNegativeTagInfos = deepNNTagInfos.clone(
	svTagInfos=cms.InputTag('pfInclusiveSecondaryVertexFinderNegativeTagInfos')
	)
deepNNNegativeTagInfos.computer.vertexFlip = True
deepNNNegativeTagInfos.computer.trackFlip = True
deepNNNegativeTagInfos.computer.trackSelection.sip3dSigMax = 0
deepNNNegativeTagInfos.computer.trackPseudoSelection.sip3dSigMax = 0
deepNNNegativeTagInfos.computer.trackPseudoSelection.sip2dSigMin = -99999.9
deepNNNegativeTagInfos.computer.trackPseudoSelection.sip2dSigMax = -2.0

negativeDeepFlavourJetTags = deepFlavourJetTags.clone(
	src=cms.InputTag('deepNNNegativeTagInfos')
	)

deepNNPositiveTagInfos = deepNNTagInfos.clone()
deepNNPositiveTagInfos.computer.trackSelection.sip3dSigMin = 0
deepNNPositiveTagInfos.computer.trackPseudoSelection.sip3dSigMin = 0
positiveDeepFlavourJetTags = deepFlavourJetTags.clone(
	src=cms.InputTag('deepNNPositiveTagInfos')
	)

# Deep CMVA
deepCMVANegativeTagInfos = deepCMVATagInfos.clone(
	deepNNTagInfos = cms.InputTag('deepNNNegativeTagInfos')
	)
	
negativeDeepFlavourCMVAJetTags = deepFlavourCMVAJetTags.clone(
	src=cms.InputTag('deepCMVANegativeTagInfos')
	)

deepCMVAPositiveTagInfos = deepCMVATagInfos.clone(
	deepNNTagInfos = cms.InputTag('deepNNPositiveTagInfos')
	)
positiveDeepFlavourCMVAJetTags = deepFlavourCMVAJetTags.clone(
	src=cms.InputTag('deepCMVAPositiveTagInfos')
	)



##
## Deep Flavour sequence, not complete as it would need the IP and SV tag infos
##
pfDeepFlavour = cms.Sequence(
	deepNNTagInfos *
	deepCMVATagInfos *
	deepFlavourJetTags *
	deepFlavourCMVAJetTags
)

