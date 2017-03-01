import FWCore.ParameterSet.Config as cms

# Default pdf for Herwig++ 2.4

herwigppPDFSettingsBlock = cms.PSet(

	# PDF for shower
	hwpp_pdf_MRST2008LO = cms.vstring(
		'+hwpp_ue_V24', 									# Default tune from 2.4, see HerwigppUE_V24
		'cp /Herwig/Partons/MRST /Herwig/Partons/cmsPDFSet',					# cmsPDFSet Default name for shower PDF
                'set /Herwig/Partons/cmsPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
                'set /Herwig/Particles/p+:PDF /Herwig/Partons/cmsPDFSet', 				# Use PDF in shower
                'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/cmsPDFSet',
	),

	# PDF for hard subprocess
	hwpp_pdf_MRST2008LO_Hard = cms.vstring(
		'+hwpp_ue_V24', 									# Default tune from 2.4, see HerwigppUE_V24
		'cp /Herwig/Partons/MRST /Herwig/Partons/cmsHardPDFSet',				# cmsHardPDFSet Default name for hard subprocess PDF
                'set /Herwig/Partons/cmsHardPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
	),
)

