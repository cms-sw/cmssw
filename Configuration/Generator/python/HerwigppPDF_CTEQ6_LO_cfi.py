import FWCore.ParameterSet.Config as cms

# CTEQ6L PDF

herwigppPDFSettingsBlock = cms.PSet(

	# PDF for shower

	# Define common block to force use with original EE5C or CMS improved CUETHS1 tune
        hwpp_pdf_CTEQ6L1_Common = cms.vstring(
                'create ThePEG::LHAPDF /Herwig/Partons/cmsPDFSet ThePEGLHAPDF.so',	# cmsPDFSet Default name for shower PDF
                'set /Herwig/Partons/cmsPDFSet:PDFName cteq6ll.LHpdf',	
                'set /Herwig/Partons/cmsPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
                'set /Herwig/Particles/p+:PDF /Herwig/Partons/cmsPDFSet',		# Use PDF in shower
                'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/cmsPDFSet',
        ),

	# Original EE5C tune
        hwpp_pdf_CTEQ6L1 = cms.vstring(
		'+hwpp_pdf_CTEQ6L1_Common',
                '+hwpp_ue_EE5C', 							# Tune for CTEQ6L1 from Herwig++ 2.7, see HerwigppUE_EE_5C 
        ),

	hwpp_pdf_CTEQ6LL = cms.vstring(
                '+hwpp_pdf_CTEQ6L1',							# Correct mixing up the name
        ),

	# CMS CUETHS1 tune based on EE5C
        hwpp_pdf_CTEQ6L1_CUETHS1 = cms.vstring(
		'+hwpp_pdf_CTEQ6L1_Common',
                '+hwpp_ue_CUETHS1', 							# Tune for CTEQ6L1 from CMS based on EE_5C see HerwigppUE_CUETHS1
        ),

	hwpp_pdf_CTEQ6LL_CUETHS1 = cms.vstring(
                '+hwpp_pdf_CTEQ6L1_CUETHS1',						# Correct mixing up the name
        ),


	# PDF for hard subprocess

	# Define common block to force use with original EE5C or CMS improved CUETHS1 tune
        hwpp_pdf_CTEQ6L1_Hard_Common = cms.vstring(
                'create ThePEG::LHAPDF /Herwig/Partons/cmsHardPDFSet ThePEGLHAPDF.so',	# cmsHardPDFSet Default name for hard subprocess PDF
                'set /Herwig/Partons/cmsHardPDFSet:PDFName cteq6ll.LHpdf',	
                'set /Herwig/Partons/cmsHardPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
        ),

	# Original EE5C tune
        hwpp_pdf_CTEQ6L1_Hard = cms.vstring(
		'+hwpp_pdf_CTEQ6L1_Hard_Common',
                '+hwpp_ue_EE5C', 							# Tune for CTEQ6L1 from Herwig++ 2.7, see HerwigppUE_EE_5C 
        ),

	hwpp_pdf_CTEQ6LL_Hard = cms.vstring(
                '+hwpp_pdf_CTEQ6L1_Hard',						# Correct mixing up the name
        ),

	# CMS CUETHS1 tune based on EE5C
        hwpp_pdf_CTEQ6L1_Hard_CUETHS1 = cms.vstring(
		'+hwpp_pdf_CTEQ6L1_Hard_Common',
                '+hwpp_ue_CUETHS1', 							# Tune for CTEQ6L1 from CMS based on EE_5C see HerwigppUE_CUETHS1 
        ),

	hwpp_pdf_CTEQ6LL_Hard_CUETHS1 = cms.vstring(
                '+hwpp_pdf_CTEQ6L1_Hard_CUETHS1',						# Correct mixing up the name
        ),
)

