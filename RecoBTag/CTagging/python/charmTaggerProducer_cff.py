import FWCore.ParameterSet.Config as cms

from RecoBTag.CTagging.charmTagsComputerCvsL_cfi import *

charmTagsComputerCvsB = charmTagsComputerCvsL.clone(
   weightFile = 'RecoBTag/CTagging/data/c_vs_b_sklearn.weight.xml',   
   variables = c_vs_b_vars_vpset
   )

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(charmTagsComputerCvsB, weightFile = 'RecoBTag/CTagging/data/c_vs_b_PhaseI.xml')

#
# Negative tagger
#

charmTagsNegativeComputerCvsL = charmTagsComputerCvsL.clone(
    slComputerCfg = dict(vertexFlip     = True,
		 	 trackFlip      = True,
		 	 SoftLeptonFlip = True,
		 	 trackSelection = dict(sip3dSigMax = 0),
		 	 trackPseudoSelection = dict(
				sip3dSigMax = 0,
				sip2dSigMin = -99999.9,
				sip2dSigMax = -2.0   )
    )
)
charmTagsNegativeComputerCvsB = charmTagsComputerCvsB.clone(
    slComputerCfg = dict(vertexFlip     = True,
		 	 trackFlip      = True,
		 	 SoftLeptonFlip = True,
		 	 trackSelection = dict(sip3dSigMax = 0),
		 	 trackPseudoSelection = dict(
				sip3dSigMax = 0,
				sip2dSigMin = -99999.9,
				sip2dSigMax = -2.0   )
    )
)
#
# Positive tagger
#

charmTagsPositiveComputerCvsL = charmTagsComputerCvsL.clone(
    slComputerCfg = dict(trackSelection = dict(sip3dSigMin = 0),
                         trackPseudoSelection = dict(sip3dSigMin = 0))
)
charmTagsPositiveComputerCvsB = charmTagsComputerCvsB.clone(
    slComputerCfg = dict(trackSelection = dict(sip3dSigMin = 0),
                         trackPseudoSelection = dict(sip3dSigMin = 0))
)
