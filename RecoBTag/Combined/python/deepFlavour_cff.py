import FWCore.ParameterSet.Config as cms
from RecoBTag.Combined.pfDeepCSVTagInfos_cfi import pfDeepCSVTagInfos
from RecoBTag.Combined.DeepCMVATagInfoProducer_cfi import pfDeepCMVATagInfos
from RecoBTag.Combined.pfDeepCSVJetTags_cfi import pfDeepCSVJetTags 
from RecoBTag.Combined.pfDeepCMVAJetTags_cfi import pfDeepCMVAJetTags

##
## Negative and positive taggers for light SF estimation
##

pfDeepCSVNegativeTagInfos = pfDeepCSVTagInfos.clone(
	svTagInfos=cms.InputTag('pfInclusiveSecondaryVertexFinderNegativeTagInfos')
	)
pfDeepCSVNegativeTagInfos.computer.vertexFlip = True
pfDeepCSVNegativeTagInfos.computer.trackFlip = True
pfDeepCSVNegativeTagInfos.computer.trackSelection.sip3dSigMax = 0
pfDeepCSVNegativeTagInfos.computer.trackPseudoSelection.sip3dSigMax = 0
pfDeepCSVNegativeTagInfos.computer.trackPseudoSelection.sip2dSigMin = -99999.9
pfDeepCSVNegativeTagInfos.computer.trackPseudoSelection.sip2dSigMax = -2.0

pfNegativeDeepCSVJetTags = pfDeepCSVJetTags.clone(
	src=cms.InputTag('pfDeepCSVNegativeTagInfos')
	)

pfDeepCSVPositiveTagInfos = pfDeepCSVTagInfos.clone()
pfDeepCSVPositiveTagInfos.computer.trackSelection.sip3dSigMin = 0
pfDeepCSVPositiveTagInfos.computer.trackPseudoSelection.sip3dSigMin = 0
pfPositiveDeepCSVJetTags = pfDeepCSVJetTags.clone(
	src=cms.InputTag('pfDeepCSVPositiveTagInfos')
	)

# Deep CMVA
pfDeepCMVANegativeTagInfos = pfDeepCMVATagInfos.clone(
	deepNNTagInfos = cms.InputTag('pfDeepCSVNegativeTagInfos')
	)
	
pfNegativeDeepCMVAJetTags = pfDeepCMVAJetTags.clone(
	src=cms.InputTag('pfDeepCMVANegativeTagInfos')
	)

pfDeepCMVAPositiveTagInfos = pfDeepCMVATagInfos.clone(
	deepNNTagInfos = cms.InputTag('pfDeepCSVPositiveTagInfos')
	)
pfPositiveDeepCMVAJetTags = pfDeepCMVAJetTags.clone(
	src=cms.InputTag('pfDeepCMVAPositiveTagInfos')
	)



##
## Deep Flavour sequence, not complete as it would need the IP and SV tag infos
##
pfDeepFlavour = cms.Sequence(
	pfDeepCSVTagInfos 
	##* pfDeepCMVATagInfos * #SKIP for the moment
	* pfDeepCSVJetTags 
	##* pfDeepCMVAJetTags
)

