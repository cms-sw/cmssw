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
    svTagInfos='pfInclusiveSecondaryVertexFinderNegativeTagInfos',
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
	src='pfDeepCSVNegativeTagInfos'
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
	src='pfDeepCSVPositiveTagInfos'
	)

# Deep CMVA
pfDeepCMVANegativeTagInfos = pfDeepCMVATagInfos.clone(
	deepNNTagInfos = 'pfDeepCSVNegativeTagInfos'
	)
	
pfNegativeDeepCMVAJetTags = pfDeepCMVAJetTags.clone(
	src='pfDeepCMVANegativeTagInfos'
	)

pfDeepCMVAPositiveTagInfos = pfDeepCMVATagInfos.clone(
	deepNNTagInfos = 'pfDeepCSVPositiveTagInfos'
	)
pfPositiveDeepCMVAJetTags = pfDeepCMVAJetTags.clone(
	src='pfDeepCMVAPositiveTagInfos'
	)



## Deep CSV+CMVA sequence, not complete as it would need the IP and SV tag infos
pfDeepCSVTask = cms.Task(
    pfDeepCSVTagInfos,
    pfDeepCMVATagInfos, #SKIP for the moment
    pfDeepCSVJetTags,
    pfDeepCMVAJetTags
)

pfDeepCSV = cms.Sequence(pfDeepCSVTask)
