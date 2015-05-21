import FWCore.ParameterSet.Config as cms

# Default pdf for Herwig++ 2.4

herwigppPDFSettingsBlock = cms.PSet(

	pdfMRST2008LOss = cms.vstring(
		'cp /Herwig/Partons/MRST cmsPDFSet',
                'cd /Herwig/Particles',
                'set p+:PDF cmsPDFSet',
                'set pbar-:PDF cmsPDFSet',
		'+ue_2_4', # Default tune from 2.4, see HerwigppUE_V24
		'cd /',
	),
)

