import FWCore.ParameterSet.Config as cms

# Default pdf for Herwig++ 2.3

herwigppPDFSettingsBlock = cms.PSet(

	pdfMRST2001 = cms.vstring(
		'cd /Herwig/Partons',
		'create Herwig::MRST MRST2001 HwMRST.so',
		'setup MRST2001 ${HERWIGPATH}/PDF/mrst/2001/lo2002.dat',
		'set MRST2001:RemnantHandler HadronRemnants',
		'cd /',
		'cp /Herwig/Partons/MRST2001 cmsPDFSet',
                'cd /Herwig/Particles',
                'set p+:PDF cmsPDFSet',
                'set pbar-:PDF cmsPDFSet',
		'+ue_2_3', # Default tune from 2.3 see HerwigppUE_V23
		'cd /',
	),
)

