import FWCore.ParameterSet.Config as cms

# Default pdf for Herwig++ 2.3

herwigppPDFSettingsBlock = cms.PSet(

	hwpp_pdf_MRST2001_Common = cms.vstring(
		'+hwpp_ue_V23', 								# Default tune from 2.3 see HerwigppUE_V23
		# /Herwig/Partons
		'create Herwig::MRST /Herwig/Partons/MRST2001 HwMRST.so',
		'setup /Herwig/Partons/MRST2001 ${HERWIGPATH}/PDF/mrst/2001/lo2002.dat',
		'set /Herwig/Partons/MRST2001:RemnantHandler /Herwig/Partons/HadronRemnants',
	),

	# PDF for shower
	hwpp_pdf_MRST2001 = cms.vstring(
		'+hwpp_pdf_MRST2001_Common',
		'cp /Herwig/Partons/MRST2001 /Herwig/Partons/cmsPDFSet',			# cmsPDFSet Default name for shower PDF
                # cd /Herwig/Particles
                'set /Herwig/Particles/p+:PDF /Herwig/Partons/cmsPDFSet', 			# Use PDF in shower
                'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/cmsPDFSet',

	),

	# PDF for hard subprocess
	hwpp_pdf_MRST2001_Hard = cms.vstring(
		'+hwpp_pdf_MRST2001_Common',
		'cp /Herwig/Partons/MRST2001 /Herwig/Partons/cmsHardPDFSet',			# cmsHardPDFSet Default name for hard subprocess PDF
	),

)

