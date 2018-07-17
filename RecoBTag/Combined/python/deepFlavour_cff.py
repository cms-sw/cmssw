import FWCore.ParameterSet.Config as cms
from RecoBTag.Combined.pfDeepCSVTagInfos_cfi import pfDeepCSVTagInfos
from RecoBTag.Combined.pfDeepCMVATagInfos_cfi import pfDeepCMVATagInfos
from RecoBTag.Combined.pfDeepCSVJetTags_cfi import pfDeepCSVJetTags 
from RecoBTag.Combined.pfDeepCSVDiscriminatorsJetTags_cfi import pfDeepCSVDiscriminatorsJetTags
from RecoBTag.Combined.pfDeepCMVAJetTags_cfi import pfDeepCMVAJetTags
from RecoBTag.Combined.pfDeepCMVADiscriminatorsJetTags_cfi import pfDeepCMVADiscriminatorsJetTags

##
## Negative and positive taggers for light SF estimation
##

pfDeepCSVNegativeTagInfos = pfDeepCSVTagInfos.clone(
    svTagInfos=cms.InputTag('pfInclusiveSecondaryVertexFinderNegativeTagInfos'),
    computer = dict(
        vertexFlip = True,
        trackFlip = True,
        trackSelection = dict( 
            sip3dSigMax = 10.0
            ),
        trackPseudoSelection = dict(
            sip3dSigMax = 10.0,
            sip2dSigMin = -99999.9,
            sip2dSigMax = -2.0
            )
        )
    )

pfNegativeDeepCSVJetTags = pfDeepCSVJetTags.clone(
	src=cms.InputTag('pfDeepCSVNegativeTagInfos')
	)

pfDeepCSVPositiveTagInfos = pfDeepCSVTagInfos.clone(
    computer = dict(
        trackSelection = dict( 
            sip3dSigMin = 0
            ),
        trackPseudoSelection = dict(
            sip3dSigMin = 0
            )
        )

    )

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



## Deep CSV+CMVA sequence, not complete as it would need the IP and SV tag infos
pfDeepCSVTask = cms.Task(
    pfDeepCSVTagInfos,
    pfDeepCMVATagInfos, #SKIP for the moment
    pfDeepCSVJetTags,
    pfDeepCMVAJetTags
)

pfDeepCSV = cms.Sequence(pfDeepCSVTask)
